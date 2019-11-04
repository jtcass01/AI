import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from FileHandler import FileHandler
from GeneticAlgorithm import GeneticAlgorithm
from Graph import Route, Graph, Edge
from WisdomOfCrowds import WisdomOfCrowds_GeneticAlgorithm, CrowdSolution

def run_tests_on_file(relative_tsp_graph_path, woc_log_location, population_size, epoch_threshold, test_log_path, superiority_tolerance=0.8):
    tests = list([])
    woc_test = TSPWOCTest(relative_tsp_graph_path=relative_tsp_graph_path,
                        population_size=population_size,
                        epoch_threshold=epoch_threshold,
                        test_log_path=test_log_path,
                        log_location=woc_log_location,
                        superiority_tolerance=superiority_tolerance)

    tests.append(woc_test)

    for cross_over_method in GeneticAlgorithm.Chromosome.CrossoverMethods:
        for mutation_method in GeneticAlgorithm.Chromosome.MutationMethods:
            if cross_over_method != GeneticAlgorithm.Chromosome.CrossoverMethods.INVALID and mutation_method != GeneticAlgorithm.Chromosome.MutationMethods.INVALID:
                ga_test = TSPGATest(relative_tsp_graph_path=relative_tsp_graph_path,
                    population_size=population_size,
    				epoch_threshold=epoch_threshold,
    				test_log_path=test_log_path,
    				cross_over_method=cross_over_method,
    				mutation_method=mutation_method)
                tests.append(ga_test)

    for test in tests:
        print("Running test:", test)
        test.run(plot=False)
        test.log()

class TSPWOCTest(object):
    def __init__(self, relative_tsp_graph_path, population_size, epoch_threshold, test_log_path, log_location, superiority_tolerance=0.8, cross_over_probability=0.8, mutation_probability=0.02):
        self.relative_tsp_graph_path = relative_tsp_graph_path
        self.population_size = population_size
        self.superiority_tolerance=superiority_tolerance
        self.cross_over_probability = cross_over_probability
        self.mutation_probability = mutation_probability
        self.epoch_threshold = epoch_threshold
        self.log_location = log_location
        self.test_name = "WOC_TEST_" + str(population_size) + "_" + str(superiority_tolerance) + "_" + str(cross_over_probability) + "_" + str(mutation_probability) + "_" + str(epoch_threshold)
        self.run_time = None
        self.result = None
        self.test_log_path = test_log_path

    def __str__(self):
        return self.test_name + ", " + str(self.run_time) + ", " + str(self.result)

    def run(self, plot = False):
        vertices = FileHandler.read_graph(os.getcwd() + os.path.sep + self.relative_tsp_graph_path, None)
        graph = Graph(vertices)
        graph.build_graph()

        start_time = time.time()

        uniform_twors = GeneticAlgorithm(graph, population_size=int(self.population_size/6), crossover_probability=self.cross_over_probability, mutation_probability=self.mutation_probability, epoch_threshold=self.epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
        uniform_rsm = GeneticAlgorithm(graph, population_size=int(self.population_size/6), crossover_probability=self.cross_over_probability, mutation_probability=self.mutation_probability, epoch_threshold=self.epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)
        partially_mapped_twors = GeneticAlgorithm(graph, population_size=int(self.population_size/6), crossover_probability=self.cross_over_probability, mutation_probability=self.mutation_probability, epoch_threshold=self.epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
        partially_mapped_rms = GeneticAlgorithm(graph, population_size=int(self.population_size/6), crossover_probability=self.cross_over_probability, mutation_probability=self.mutation_probability, epoch_threshold=self.epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)
        ordered_crossover_twors = GeneticAlgorithm(graph, population_size=int(self.population_size/6), crossover_probability=self.cross_over_probability, mutation_probability=self.mutation_probability, epoch_threshold=self.epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
        ordered_crossover_rsm = GeneticAlgorithm(graph, population_size=int(self.population_size/6), crossover_probability=self.cross_over_probability, mutation_probability=self.mutation_probability, epoch_threshold=self.epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)

        algorithms = [uniform_twors, uniform_rsm, partially_mapped_twors, partially_mapped_rms, ordered_crossover_twors, ordered_crossover_rsm]

        test_algorithm = WisdomOfCrowds_GeneticAlgorithm(genetic_algorithms=algorithms, weights=[0.05, 0.05, 0.05, 0.05, 0.6, 0.2], log_location=self.log_location)
        test_algorithm.run()
        test_crowd_solution = CrowdSolution(graph)
        test_crowd_solution.load(self.log_location)
#        test_crowd_solution.generate_heat_map(superiority_tolerance=self.superiority_tolerance)
        test_crowd_solution.complete_graph_greedy_heuristic(superiority_tolerance=self.superiority_tolerance)

        if plot:
            test_crowd_solution.route.plot()
        self.result = test_crowd_solution.route.distance_traveled

        self.run_time = time.time() - start_time

    def log(self):
        with open(self.test_log_path, "a+") as log_file:
            log_file.write(str(self) + "\n")


class TSPGATest(object):
    def __init__(self, relative_tsp_graph_path, population_size, cross_over_method, mutation_method, epoch_threshold, test_log_path, cross_over_probability=0.8, mutation_probability=0.02):
        self.relative_tsp_graph_path = relative_tsp_graph_path
        self.population_size = population_size
        self.cross_over_method = cross_over_method
        self.mutation_method = mutation_method
        self.cross_over_probability = cross_over_probability
        self.mutation_probability = mutation_probability
        self.epoch_threshold = epoch_threshold
        self.test_name = "GA_TEST_" + str(cross_over_method) + str(cross_over_probability) + "_" + str(mutation_method) + str(mutation_probability) + "_" + str(epoch_threshold)
        self.run_time = None
        self.result = None
        self.test_log_path = test_log_path

    def __str__(self):
        return self.test_name + ", " + str(self.run_time) + ", " + str(self.result)

    def run(self, plot = False):
        vertices = FileHandler.read_graph(os.getcwd() + os.path.sep + self.relative_tsp_graph_path, None)
        graph = Graph(vertices)
        graph.build_graph()

        start_time = time.time()

        test_algorithm = GeneticAlgorithm(graph=graph, population_size=self.population_size, crossover_probability=self.cross_over_probability, mutation_probability=self.mutation_probability, epoch_threshold=self.epoch_threshold, crossover_method=self.cross_over_method, mutation_method=self.mutation_method)
        resultant_route = test_algorithm.run()

        if plot:
            resultant_route.plot()

        self.result = resultant_route.distance_traveled

        self.run_time = time.time() - start_time

    def log(self):
        with open(self.test_log_path, "a+") as log_file:
            log_file.write(str(self) + "\n")


if __name__ == "__main__":
    tsp_file_endings = ["Random77", "Random97", "Random222"]

    for tsp_file_ending in tsp_file_endings:
        test_log_path = os.getcwd() + os.path.sep + ".." + os.path.sep + "results" + os.path.sep + "TEST_RESULTS_" + tsp_file_ending + "_" + datetime.datetime.now().isoformat()[:10] + "_0.csv"
        woc_log_location = os.getcwd() + os.path.sep + ".." + os.path.sep + "results" + os.path.sep + "crowd_" + tsp_file_ending + "_" + datetime.datetime.now().isoformat()[:10] + "_0.csv"
        run_tests_on_file(relative_tsp_graph_path=".." + os.path.sep + "docs" + os.path.sep + "datasets" + os.path.sep + tsp_file_ending + ".tsp",
        population_size=300,
        epoch_threshold=25,
        test_log_path=test_log_path,
        woc_log_location=woc_log_location,
        superiority_tolerance=0.8)
