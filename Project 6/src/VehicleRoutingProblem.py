#!/usr/bin/python
import os
import random
import numpy as np
import pandas as pd

from Graph import Graph
from FileHandler import FileHandler
from GeneticAlgorithm import VRP_GeneticAlgorithm

class VehicleRoutingProblem(object):
    def __init__(self, algorithm, graph, number_of_depots=1, customers_per_depot=1):
        assert number_of_depots > 0 and number_of_depots < len(graph.vertices), "The number of depots must be greater than or equal to 1 and less than the number of vertices."
        assert customers_per_depot > 0 and number_of_depots < len(graph.vertices), "The number of customers per depot must be greater than or equal to 1 and less than the number of vertices."
        self.algorithm = algorithm
        self.graph = graph
        self.depot_locations = VehicleRoutingProblem.generate_random_depot_locations(graph, number_of_depots)
        self.customers_df = VehicleRoutingProblem.generate_random_customers(graph, self.depot_locations, customers_per_depot)

    def run(self):
        for depot_location, customers_row in zip(self.depot_locations, self.customers_df.iterrows()):
            customers = customers_row[1]
            self.algorithm.initialize_population(depot_location, customers)
            self.algorithm.run()

    @staticmethod
    def generate_random_depot_locations(graph, number_of_depots):
        return np.array(random.sample(list(graph.vertices), number_of_depots))

    @staticmethod
    def generate_random_customers(graph, depot_locations, customers_per_depot):
        def generate_random_customers_for_depot_location(depot_location):
            return np.array(random.sample(list(graph.vertices[np.where(graph.vertices != depot_location)]), customers_per_depot))
        random_customers = {}

        for depot_location in depot_locations:
            random_customers[depot_location.vertex_id] = generate_random_customers_for_depot_location(depot_location)

        random_customers_df = pd.DataFrame.from_dict(random_customers, orient='index')
        return random_customers_df


def vehicle_routing_problem_test(graph_location):
    # Read in the test data
    graph = Graph(FileHandler.read_graph(graph_location))

    test_algorithm = VRP_GeneticAlgorithm(graph=graph, population_size=50, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=30, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    test_problem = VehicleRoutingProblem(test_algorithm, graph, 3, 4)
    test_problem.run()

if __name__ == "__main__":
    vehicle_routing_problem_test(graph_location=os.getcwd() + os.path.sep + ".." + os.path.sep + "datasets" + os.path.sep + "Random44.tsp")
