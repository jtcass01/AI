import os

def solve_traveling_salesman(relative_tsp_py_path, relative_tsp_graph_path, relative_adj_graph_path):
	cwd = os.getcwd()

	tsp_py_path = "\"" + cwd + relative_tsp_py_path + "\""

	tsp_graph_path = relative_tsp_graph_path

	adj_graph_path = relative_adj_graph_path

	system_call = "python {} BFS {} {}".format(tsp_py_path, tsp_graph_path, adj_graph_path)

	print(system_call)

	os.system(system_call)

if __name__ == "__main__":
	solve_traveling_salesman(os.path.sep + "TravelingSalesman.py",
							 ".." + os.path.sep + "docs" + os.path.sep + "11PointDFSBFS.tsp",
							 ".." + os.path.sep + "docs" + os.path.sep + "AdjacencyMatrix.csv")
