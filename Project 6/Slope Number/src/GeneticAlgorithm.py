#!/usr/bin/python
import numpy as np
import pandas as pd
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
        states = [1] * int(len(self.graph.vertices) * 0.05) + [0] * int(len(self.graph.vertices) * 0.95)
        connection_dictionary = {}
        if len(states) < len(self.graph.vertices):
            states += [1] * (len(self.graph.vertices)-len(states))

        for chromosome_index in range(self.population_size):
            for row_index in range(1, len(self.graph.vertices)+1):
                connection_dictionary[str(row_index)] = random.sample(states, len(self.graph.vertices))
                # Assert there are no self connections.
                connection_dictionary[str(row_index)][row_index-1] = 0

            dataframe = pd.DataFrame.from_dict(connection_dictionary, orient='index', columns=list(range(1, len(self.graph.vertices)+1)))
            random_route = Route(self.graph)
            random_route.load_edge_dataframe(dataframe)
            chromosome = GeneticAlgorithm.Chromosome(chromosome_index, route=random_route, allele_dataframe=dataframe, crossover_method=self.crossover_method, mutation_method=self.mutation_method)
            self.population.append(chromosome)

        self.population = np.array(self.population)

    def run(self):
        improvement = 0
        self.costs = list([])
        epochs_since_last_improvement = 0
        best_chromosome = min(self.population)
        self.best_chromosome = best_chromosome
        self.costs.append(best_chromosome.route.slope_number)

        while epochs_since_last_improvement < self.epoch_threshold:
            print("performing crossovers")
            # Perform cross overs
            self.perform_crossovers()

            print("performing mutations")
            # Perform mutations
            self.perform_mutations()

            # Get new best_chromosome
            best_chromosome = min(self.population)

            improvement = self.best_chromosome.route.slope_number - best_chromosome.route.slope_number
            print("improvement", improvement, "epochs since last improvement", epochs_since_last_improvement)

            if improvement > 0:
                self.best_chromosome = best_chromosome
                epochs_since_last_improvement = 0
            else:
                epochs_since_last_improvement += 1

            self.costs.append(self.best_chromosome.route.slope_number)

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
        def __init__(self, chromosome_id, route, allele_dataframe, crossover_method, mutation_method):
            self.chromosome_id = chromosome_id
            self.route = route
            self.allele_dataframe = allele_dataframe
            self.crossover_method = crossover_method
            self.mutation_method = mutation_method

        def __str__(self):
            return "Chromosome #"  + str(self.chromosome_id) + " | Slope #" + str(self.route.slope_number)

        def display_vertex_ids(self):
            string = "["
            for vertex in self.route.vertices:
                string += str(vertex.vertex_id) + ", "

            print(string[:-2] + "]")

        def retrieve_alleles(self):
            alleles = None

            for row_index, row in self.allele_dataframe.iterrows():
                if alleles is None:
                    alleles = np.array(list(row[:int(row_index)-1]))
                    alleles = np.concatenate((alleles, list(row[int(row_index):])), axis=None)
                else:
                    alleles = np.concatenate((alleles, list(row[:int(row_index)-1])), axis=None)
                    alleles = np.concatenate((alleles, list(row[int(row_index):])), axis=None)

            return alleles.reshape((1, -1))

        def update_allele_dataframe(self, alleles):
            allele_index = 0
            connection_dictionary = {}

            for row_index, row in self.allele_dataframe.iterrows():
                for column_index, connection_boolean in enumerate(row):
                    if int(row_index) != column_index+1:
                        if str(row_index) not in connection_dictionary:
                            connection_dictionary[row_index] = np.array([alleles[0, allele_index]])
                        else:
                            connection_dictionary[row_index] = np.append(connection_dictionary[row_index], [alleles[0, allele_index]])
                        allele_index += 1
                    else:
                        if str(row_index) not in connection_dictionary:
                            connection_dictionary[row_index] = np.array([0])
                        else:
                            connection_dictionary[row_index] = np.append(connection_dictionary[row_index], [0])
            dataframe = pd.DataFrame.from_dict(connection_dictionary, orient='index', columns=list(range(1, len(self.route.graph.vertices)+1)))
            self.allele_dataframe.update(dataframe)
            self.route.load_edge_dataframe(self.allele_dataframe)

        def crossover(self, other_chromosome):
            new_alleles = list([])
            self_alleles = self.retrieve_alleles()
            other_alleles = other_chromosome.retrieve_alleles()

            if self.crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM:
                alleles = np.concatenate((self_alleles[0, :], other_alleles[0, :]), axis=None)
                new_alleles = np.array(random.sample(list(alleles), len(self_alleles[0, :]))).reshape((1, -1))

            elif self.crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED:
                p1 = random.randint(1, len(self_alleles[0, :])-2)
                p2 = random.randint(p1+1, len(self_alleles[0, :])-1)

                self_s1 = self_alleles[0, :p1]
                self_s2 = self_alleles[0, p1:p2]
                self_s3 = self_alleles[0, p2:]

                other_s1 = other_alleles[0, :p1]
                other_s2 = other_alleles[0, p1:p2]
                other_s3 = other_alleles[0, p2:]

                new_alleles = self_s1

                s2_left = list([])
                for allele in other_s2:
                    if allele not in self_s1 and allele not in self_s3:
                        s2_left.append(allele)

                s3_left = list([])
                for allele in other_s3:
                    if allele not in self_s1 and allele not in self_s3:
                        s3_left.append(allele)

                s1_left = list([])
                for allele in other_s1:
                    if allele not in self_s1 and allele not in self_s3:
                        s1_left.append(allele)

                remaining_alleles = s2_left + s3_left + s1_left
                new_alleles += remaining_alleles[:p2-p1]

                new_alleles += self_s3

            elif self.crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER:
                p1 = random.randint(1, len(self_alleles[0, :])-2)
                p2 = random.randint(p1+1, len(self_alleles[0, :])-1)
                added_indeces = list([])
                j_1 = p1 + 1
                j_2 = j_1
                k = j_1

                to_p1 = self_alleles[0, :p1]
                to_p1_indices = list(range(0,p1))
                from_p1 = self_alleles[0, p1:]
                from_p1_indices = list(range(p1, len(self_alleles[0, :])))
                mid = other_alleles[0, p1:p2+1]
                mid_indices = list(range(p1, p2+1))

#                print("to_p1", to_p1, list(to_p1_indices))
#                print("from_p1", from_p1, list(from_p1_indices))
#                print("mid", mid, list(mid_indices))

                for allele_index, allele in zip(to_p1_indices, to_p1):
                    if allele_index not in added_indeces:
                        new_alleles.append(allele)
                        added_indeces.append(allele_index)

                for allele_index, allele in zip(mid_indices, mid):
                    if allele_index not in added_indeces:
                        new_alleles.append(allele)
                        added_indeces.append(allele_index)

                for allele_index, allele in zip(from_p1_indices, from_p1):
                    if allele_index not in added_indeces:
                        new_alleles.append(allele)
                        added_indeces.append(allele_index)

                new_alleles = np.array(new_alleles).reshape((1, -1))

#            print("self_alleles", self_alleles, self_alleles.shape)
#            print("other_alleles", other_alleles, other_alleles.shape)
#            print("new_alleles", new_alleles, new_alleles.shape)
            resultant_chromosome = deepcopy(self)
            resultant_chromosome.update_allele_dataframe(new_alleles)

            return resultant_chromosome

        def mutate(self):
            new_alleles = list([])
            self_alleles = self.retrieve_alleles()

            if self.mutation_method == GeneticAlgorithm.Chromosome.MutationMethods.TWORS:
                # Generate random indices for swapping
                mutated_index_0 = random.randint(0, len(self_alleles)-2)
                mutated_index_1 = random.randint(mutated_index_0+1, len(self_alleles)-1)
                swap_vertex = None

                # Iterate over the vertices until the swap_vertex is found.  Keep track and replace when at new location.
                for allele_index, allele in enumerate(self_alleles):
                    if allele_index == mutated_index_0:
                        swap_allele = allele
                    elif allele_index == mutated_index_1:
                        new_alleles.append(swap_allele)
                        new_alleles.insert(mutated_index_0, allele)
                    else:
                        new_alleles.append(allele)
            elif self.mutation_method == GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION:
                for allele in np.flip(self_alleles):
                    new_alleles.append(allele)

            # Cast to NumPy Array.  Reset route and walk the new path.
            new_alleles = np.array(new_alleles)
            self.update_allele_dataframe(new_alleles)

        def __eq__(self, other):
            return self.route.slope_number == other.route.slope_number

        def __lt__(self, other):
            return self.route.slope_number < other.route.slope_number

        def __le__(self, other):
            return self.route.slope_number <= other.route.slope_number

        def __gt__(self, other):
            return self.route.slope_number > other.route.slope_number

        def __ge__(self, other):
            return self.route.slope_number >= other.route.slope_number

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

            parent_1 = [1, 2, 3, 4, 5, 6]
            print("Parent 1: ", parent_path_1)
            parent_chromosome_1 = build_chromosome_from_path_and_graph(1, parent_path_1, graph, crossover_method, mutation_method)

            parent_2 = [6, 5, 4, 3, 2, 1]
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
    print("loading graph")
    # Read in test data
    graph = Graph(FileHandler.read_graph(os.getcwd() + os.path.sep + ".." + os.path.sep + "datasets" + os.path.sep + "Random97.tsp"))

    print("Initializing genetic algorithm")
    test_algorithm = GeneticAlgorithm(graph=graph, population_size=50, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=50, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)

    print("running test algorithm")
    test_algorithm.run()

    print("Displaying results")
    test_algorithm.display_result()

if __name__ == "__main__":
    slope_number_test()
