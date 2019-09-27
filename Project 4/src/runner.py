import os

def test_display(relative_display_path, relative_tsp_graph_path):
	cwd = os.getcwd()

	display_path = "\"" + cwd + relative_display_path + "\""

	tsp_graph_path = relative_tsp_graph_path

	system_call = "python {} {}".format(display_path, tsp_graph_path)

	print(system_call)

	os.system(system_call)

def test_genetic_algorithm(relative_tsp_path, relative_tsp_graph_path):
	cwd = os.getcwd()

	tsp_path = "\"" + cwd + os.path.sep + relative_tsp_path + "\""

	tsp_graph_path = relative_tsp_graph_path

	system_call = "python {} {} {} {}".format(tsp_path, "genetic", tsp_graph_path, "none")

	print(system_call)

	os.system(system_call)

if __name__ == "__main__":
	test_genetic_algorithm("TravelingSalesman.py", ".." + os.path.sep + "docs" + os.path.sep + "Random100.tsp")
