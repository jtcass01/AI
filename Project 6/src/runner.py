import os
import datetime
import itertools

def run_full_tests():
	datasets = list(["Random6", "Random7", "Random8", "Random11", "Random22", "Random44", "Random77", "Random97", "Random222"])
	dataset_sizes = list([6, 7, 8, 11, 22, 44, 77, 97, 222])
	assert len(datasets) == len(dataset_sizes)

	for dataset, dataset_size in zip(datasets, dataset_sizes):
		population_sizes_per_genetic_algorithm = [5, 10, 25, 50, 75]
		epoch_thresholds = [10, 25, 50, 75, 100]
		crossover_probabilities = [0.2, 0.4, 0.6, 0.8]
		mutation_probabilities = [0.01, 0.1, 0.25, 0.5]
		number_of_depots_possibilities = list(range(1, dataset_size))
		number_of_customers_possibilities = list(range(int(dataset_size/2), dataset_size-1))
		pop_epoch_crossover_mutation = list(itertools.product(population_sizes_per_genetic_algorithm, epoch_thresholds, crossover_probabilities, mutation_probabilities, number_of_depots_possibilities, number_of_customers_possibilities))

		for population_size_per_genetic_algorithm, epoch_threshold, crossover_probability, mutation_probability, number_of_depots, number_of_customers in pop_epoch_crossover_mutation:
			system_call = "python VehicleRoutingProblem.py {} {} {} {} {} {} {}".format(dataset, population_size_per_genetic_algorithm, epoch_threshold, crossover_probability, mutation_probability, number_of_depots, number_of_customers)
			print(system_call)
			os.system(system_call)


if __name__ == "__main__":
	run_full_tests()
