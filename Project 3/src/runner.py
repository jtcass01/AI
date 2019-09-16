import os

def test_display(relative_display_path, relative_tsp_graph_path):
	cwd = os.getcwd()

	display_path = "\"" + cwd + relative_display_path + "\""

	tsp_graph_path = relative_tsp_graph_path

	system_call = "python {} {}".format(display_path, tsp_graph_path)

	print(system_call)

	os.system(system_call)

if __name__ == "__main__":
	test_display(os.path.sep + "Display.py",
                ".." + os.path.sep + "docs" + os.path.sep + "Random40.tsp")
