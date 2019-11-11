#!/usr/bin/python
import os
import random
import datetime
import time
import numpy as np
import pandas as pd
from copy import deepcopy
import sys

from Graph import Graph
from FileHandler import FileHandler
from GeneticAlgorithm import VRP_GeneticAlgorithm
from WisdomOfCrowds import WisdomOfCrowds_GeneticAlgorithm

class VehicleRoutingProblem(object):
    def __init__(self, algorithm, graph, number_of_depots=1, customers_per_depot=1):
        assert number_of_depots > 0 and number_of_depots <= len(graph.vertices), "The number of depots must be greater than or equal to 1 and less than the number of vertices."
        assert customers_per_depot > 0 and number_of_depots < len(graph.vertices), "The number of customers per depot must be greater than or equal to 1 and less than the number of vertices."
        self.algorithm = algorithm
        self.graph = graph
        self.depot_locations = VehicleRoutingProblem.generate_random_depot_locations(graph, number_of_depots)
        self.customers_df = VehicleRoutingProblem.generate_random_customers(graph, self.depot_locations, customers_per_depot)

    def run(self):
        total_cost = 0

        for depot_location, customers_row in zip(self.depot_locations, self.customers_df.iterrows()):
            depot_algorithm = deepcopy(self.algorithm)
            customers = customers_row[1]
            depot_algorithm.initialize_population(depot_location, customers)
            depot_algorithm.run()
            for vertex in self.graph.vertices:
                if isinstance(depot_algorithm, WisdomOfCrowds_GeneticAlgorithm):
                    if vertex not in depot_algorithm.crowd_solution.route.vertices:
                        depot_algorithm.crowd_solution.route.vertices = np.append(depot_algorithm.crowd_solution.route.vertices, [vertex])
                elif isinstance(depot_algorithm, VRP_GeneticAlgorithm):
                    if vertex not in depot_algorithm.best_chromosome.route.vertices:
                        depot_algorithm.best_chromosome.route.vertices = np.append(depot_algorithm.best_chromosome.route.vertices, [vertex])

            total_cost += depot_algorithm.get_cost()

        return total_cost

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


def vehicle_routing_problem_test(dataset, population_size_per_genetic_algorithm, epoch_threshold, crossover_probability, mutation_probability, number_of_depots, number_of_customers):
    algorithms = list([])

    graph_location = os.getcwd() + os.path.sep + ".." + os.path.sep + "datasets" + os.path.sep + dataset +".tsp"
    log_location=os.getcwd() + os.path.sep + ".." + os.path.sep + "results" + os.path.sep + "crowd_" + dataset +"_" + datetime.datetime.now().isoformat()[:10] + "_0.csv"
    test_data_labels = list(["population_size_per_genetic_algorithm", "epoch_threshold", "crossover_probability", "mutation_probability", "number_of_depots", "number_of_customers"])
    test_data = list([population_size_per_genetic_algorithm, epoch_threshold, crossover_probability, mutation_probability, number_of_depots, number_of_customers])

    graph = Graph(FileHandler.read_graph(graph_location))

    uniform_twors = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    uniform_rsm = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)
    partially_mapped_twors = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    partially_mapped_rms = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)
    ordered_crossover_twors = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    ordered_crossover_rsm = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)

    algorithms = [uniform_twors, uniform_rsm, partially_mapped_twors, partially_mapped_rms, ordered_crossover_twors, ordered_crossover_rsm]

    woc_uniform_twors = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    woc_uniform_rsm = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)
    woc_partially_mapped_twors = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    woc_partially_mapped_rms = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)
    woc_ordered_crossover_twors = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    woc_ordered_crossover_rsm = VRP_GeneticAlgorithm(graph, population_size=population_size_per_genetic_algorithm, crossover_probability=crossover_probability, mutation_probability=mutation_probability, epoch_threshold=epoch_threshold, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)
    woc_algorthms = list([woc_uniform_twors, woc_uniform_rsm, woc_partially_mapped_twors, woc_partially_mapped_rms, woc_ordered_crossover_twors, woc_ordered_crossover_rsm])
    woc_algorithm = WisdomOfCrowds_GeneticAlgorithm(genetic_algorithms=woc_algorthms, weights=[0.05, 0.05, 0.05, 0.05, 0.6, 0.2], log_location=log_location)
    algorithms.append(woc_algorithm)

#    test_log_location = os.getcwd() + os.path.sep + ".." + os.path.sep + "results" + os.path.sep + "VehicleRoutingProblemTest_" + dataset +"_" + str(number_of_depots) + "_" +  str(number_of_customers) + "_" + datetime.datetime.now().isoformat()[:10] + ".csv"
    test_log_location = os.getcwd() + os.path.sep + ".." + os.path.sep + "results" + os.path.sep + "VehicleRoutingProblemTest_" + dataset +"_" + str(number_of_depots) + "_" +  str(number_of_customers) + "_" + "2019-11-10" + ".csv"
    FileHandler.start_test(test_log_location, data_labels=test_data_labels)

    distances_traveled = list([])
    run_times = list([])
    for algorithm in algorithms:
        test_problem = VehicleRoutingProblem(algorithm, graph, number_of_depots, number_of_customers)

        start = time.time()
        distance_traveled = test_problem.run()
        run_time = time.time() - start

        FileHandler.log_test(test_log_location, test_name=str(algorithm), test_result=distance_traveled, test_runtime=run_time, test_data=test_data)

        distances_traveled.append(distance_traveled)
        run_times.append(run_time)

    average_distance_traveled = sum(distances_traveled) / len(distances_traveled)
    average_run_time = sum(run_times) / len(run_times)
    FileHandler.log_test(test_log_location, test_name="AVERAGE_FOR_ALGORITHMS", test_result=average_distance_traveled, test_runtime=average_run_time, test_data=test_data)


if __name__ == "__main__":
    dataset, population_size_per_genetic_algorithm, epoch_threshold, crossover_probability, mutation_probability, number_of_depots, number_of_customers = sys.argv[1:]
    vehicle_routing_problem_test(dataset, int(population_size_per_genetic_algorithm), int(epoch_threshold), float(crossover_probability), float(mutation_probability), int(number_of_depots), int(number_of_customers))
