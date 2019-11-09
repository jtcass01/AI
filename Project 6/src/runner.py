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

	print("relative_tsp_graph_path",relative_tsp_graph_path)

	system_call = "python {} {} {}".format(tsp_path, "genetic", relative_tsp_graph_path)

	print(system_call)

	os.system(system_call)

def test_woc_algorithm(relative_tsp_path, relative_tsp_graph_path):
	cwd = os.getcwd()

	tsp_path = "\"" + cwd + os.path.sep + relative_tsp_path + "\""

	tsp_graph_path = relative_tsp_graph_path

	print("relative_tsp_graph_path",relative_tsp_graph_path)

	system_call = "python {} {} {}".format(tsp_path, "woc", relative_tsp_graph_path)

	print(system_call)

	os.system(system_call)

def run_full_tests():
	cwd = os.getcwd()
	test_path = "\"" + cwd + os.path.sep + "Test.py" + "\""

	system_call = "python {}".format(test_path)
	print(system_call)
	os.system(system_call)

if __name__ == "__main__":
	test_woc_algorithm(relative_tsp_path="TravelingSalesman.py", relative_tsp_graph_path=os.path.sep + ".." + os.path.sep + "docs" + os.path.sep + "datasets" + os.path.sep + "Random44.tsp")
#	for epoch in range(15):
#		run_full_tests()
