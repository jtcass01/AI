import os
import datetime
import itertools

def run_full_tests():
	datasets = list(["Random22"])
	dataset_sizes = list([22])
	assert len(datasets) == len(dataset_sizes)

	for dataset, dataset_size in zip(datasets, dataset_sizes):
		population_sizes_per_genetic_algorithm = [25, 50, 10]
		epoch_thresholds = [25]
		crossover_probabilities = [0.6, 0.8]
		mutation_probabilities = [0.1, 0.25, 0.5]
		number_of_depots_possibilities = [1]
		number_of_customers_possibilities = [20]
		pop_epoch_crossover_mutation = list(itertools.product(population_sizes_per_genetic_algorithm, epoch_thresholds, crossover_probabilities, mutation_probabilities, number_of_depots_possibilities, number_of_customers_possibilities))

		for population_size_per_genetic_algorithm, epoch_threshold, crossover_probability, mutation_probability, number_of_depots, number_of_customers in pop_epoch_crossover_mutation:
			system_call = "python VehicleRoutingProblem.py {} {} {} {} {} {} {}".format(dataset, population_size_per_genetic_algorithm, epoch_threshold, crossover_probability, mutation_probability, number_of_depots, number_of_customers)
			print(system_call)
			os.system(system_call)


if __name__ == "__main__":
	for test_run in range(10):
		run_full_tests()
