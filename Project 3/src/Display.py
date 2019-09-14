import sys
import os
import time

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from FileHandler import FileHandler
from Graph import Graph, Route
from TravelingSalesman import TravelingSalesman

class TravelingSalesmanGUI(QMainWindow):
    def __init__(self, algorithm):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Traveling Salesman Problem'
        self.width = 940
        self.height = 640
        self.algorithm = algorithm
        self.problem_display = None
        self.step_forward_button = None
        self.step_backward_button = None
        self.run_simulation_button = None
        self.show_all_edges_button = None
        self.initUI()

        self.reference = 1


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.problem_display = PlotCanvas(self, width=8, height=6.4)
        self.problem_display.move(0,0)

        self.step_forward_button = QPushButton('Step Forward', self)
        self.step_forward_button.setToolTip('Increase number of cities')
        self.step_forward_button.move(800,0)
        self.step_forward_button.resize(140,100)
        self.step_forward_button.clicked.connect(self.step_forward)

        self.step_backward_button = QPushButton('Step Backward', self)
        self.step_backward_button.setToolTip('Decrease number of cities')
        self.step_backward_button.move(800,100)
        self.step_backward_button.resize(140,100)
        self.step_backward_button.clicked.connect(self.step_backward)

        self.run_simulation_button = QPushButton('Run Simulation', self)
        self.run_simulation_button.setToolTip('Run simultion until completion while displaying each step.')
        self.run_simulation_button.move(800,200)
        self.run_simulation_button.resize(140,100)
        self.run_simulation_button.clicked.connect(self.run_simulation)

        self.finish_simulation_button = QPushButton('Finish Simulation', self)
        self.finish_simulation_button.setToolTip('Run simultion until completion.')
        self.finish_simulation_button.move(800,300)
        self.finish_simulation_button.resize(140,100)
        self.finish_simulation_button.clicked.connect(self.finish_simulation)

        self.show_all_edges_button = QPushButton('Show All Edges', self)
        self.show_all_edges_button.setToolTip('Toggle to show all edges.')
        self.show_all_edges_button.move(800,400)
        self.show_all_edges_button.resize(140,100)
        self.show_all_edges_button.clicked.connect(lambda: self.show_all_edges(self.show_all_edges_button))

        self.show()

    @pyqtSlot()
    def step_forward(self):
        if not self.algorithm.done:
            self.algorithm.step_forward()

            self.problem_display.plot()

    @pyqtSlot()
    def step_backward(self):
        self.algorithm.step_backward()

        self.problem_display.plot()

    @pyqtSlot()
    def run_simulation(self):
        while not self.algorithm.done:
            self.algorithm.step_forward()
            self.problem_display.plot()
            time.sleep(1)

    @pyqtSlot()
    def finish_simulation(self):
        while not self.algorithm.done:
            self.algorithm.step_forward()
        self.problem_display.plot()

    @pyqtSlot()
    def show_all_edges(self, button):
        if button.text() == 'Show All Edges':
            self.show_all_edges_button.setText('Show Current Route')
            self.show_all_edges_button.setToolTip('Toggle to show the current route.')
            self.problem_display.show_all_edges()
        else:
            self.show_all_edges_button.setText('Show All Edges')
            self.show_all_edges_button.setToolTip('Toggle to show all edges.')
            self.problem_display.plot()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.parent = parent
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):
        self.axes.clear()

        visited_vertex_x = list([])
        visited_vertex_y = list([])

        unvisited_vertex_x = list([])
        unvisited_vertex_y = list([])

        plots = list([])
        arrow_plots = list([])
        arrow_labels = list([])
        route_order = self.parent.algorithm.route.vertex_order
        graph_vertices = self.parent.algorithm.route.graph.vertices

        visited_vertices = [vertex for vertex in graph_vertices if vertex.vertex_id in route_order]
        unvisited_vertices = [vertex for vertex in graph_vertices if vertex not in visited_vertices]

        # Iterate over vertices, retrieving x and y coordinates
        for vertex in visited_vertices:
            visited_vertex_x.append(vertex.x)
            visited_vertex_y.append(vertex.y)

        for vertex in unvisited_vertices:
            unvisited_vertex_x.append(vertex.x)
            unvisited_vertex_y.append(vertex.y)

        # Plot the vertices
        ax = self.figure.add_subplot(111)

        visited_vertex_plot = ax.scatter(visited_vertex_x, visited_vertex_y, label="Visited Vertices", color='blue')
        plots.append(visited_vertex_plot)

        unvisited_vertex_plot = ax.scatter(unvisited_vertex_x, unvisited_vertex_y, label="Unvisited Vertices", color='black')
        plots.append(unvisited_vertex_plot)

        # Plot the edges
        for vertex_index in range(len(route_order)-1):
            arrow_label = "Edge {}->{}".format(route_order[vertex_index], route_order[vertex_index+1])
            arrow_plot = ax.arrow(graph_vertices[route_order[vertex_index]-1].x,
                                   graph_vertices[route_order[vertex_index]-1].y,
                                   graph_vertices[route_order[vertex_index+1]-1].x - graph_vertices[route_order[vertex_index]-1].x,
                                   graph_vertices[route_order[vertex_index+1]-1].y - graph_vertices[route_order[vertex_index]-1].y,
                                   head_width=1, head_length=1,
                                   color='#{}{}{}'.format(Graph.color_quantization(graph_vertices[route_order[vertex_index]-1].vertex_id, len(graph_vertices)),
                                                          Graph.color_quantization(graph_vertices[route_order[vertex_index]-1].vertex_id % graph_vertices[route_order[vertex_index+1]-1].vertex_id, len(graph_vertices)),
                                                          Graph.color_quantization(graph_vertices[route_order[vertex_index+1]-1].vertex_id, len(graph_vertices))),
                                   label=arrow_label)
            arrow_labels.append(arrow_label)
            arrow_plots.append(arrow_plot)

        # Show the graph with a legend
        ax.set_title('Program State')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(arrow_plots, arrow_labels, loc=2, fontsize='small')

        self.draw()

    def show_all_edges(self):
        self.axes.clear()

        x = list([])
        y = list([])
        plots = list([])
        arrow_plots = list([])
        arrow_labels = list([])

        # Iterate over vertices, retrieving x and y coordinates
        for vertex in self.parent.graph.vertices:
            x.append(vertex.x)
            y.append(vertex.y)

        # Plot the vertices
        ax = self.figure.add_subplot(111)
        vertex_plot = ax.scatter(x, y, label="Vertices")
        plots.append(vertex_plot)

        # Plot the edges
        for vertex in self.parent.graph.vertices:
            for adjacent_vertex in vertex.adjacent_vertices:
                arrow_label = "Edge {}->{}".format(vertex.vertex_id, adjacent_vertex.vertex_id)
                arrow_plot = ax.arrow(vertex.x, vertex.y, adjacent_vertex.x-vertex.x, adjacent_vertex.y-vertex.y,
                                       head_width=1, head_length=1,
                                       color='#{}{}{}'.format(Graph.color_quantization(vertex.vertex_id, len(self.parent.graph.vertices)),
                                                              "00",
                                                              Graph.color_quantization(adjacent_vertex.vertex_id, len(self.parent.graph.vertices))),
                                       label=arrow_label)

                plots.append(arrow_plot)
                arrow_plots.append(arrow_plot)
                arrow_labels.append(arrow_label)

        # Show the graph with a legend
        ax.set_title('Program State')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(arrow_plots, arrow_labels, loc=2, fontsize='small')

        self.draw()


if __name__ == '__main__':
    # Retrieve command line arguments
    if len(sys.argv) != 3 and len(sys.argv) != 2:
        print("Command Line Arguments should follow the format:")
        print("python TrainingSalesman.py [relative path to vertex_graph_file] [optional: relative path to adjacency_matrix_file]")
    else:
        # retrieve relative path to vertex_graph_file
        vertex_graph_file_path = sys.argv[1]

        adjacency_matrix = None
        if len(sys.argv) == 3:
            # retrieve relative path to adjacency_matrix_file_path
            adjacency_matrix_file_path = sys.argv[2]

            # Read the adjacency matrix
            adjacency_matrix = FileHandler.read_adjacency_matrix(os.getcwd() + os.path.sep + adjacency_matrix_file_path)

        # Read the vertices from the vertex graph file.
        vertices = FileHandler.read_graph(os.getcwd() + os.path.sep + vertex_graph_file_path, adjacency_matrix)

        # Build a graph representing the vertices and edges.
        graph = Graph(vertices)

        # Calculate edges
        graph.build_graph()

        # Start the GUI
        app = QApplication(sys.argv)
        ex = TravelingSalesmanGUI(TravelingSalesman.GreedyAlgorithm(Route([], graph)))
        sys.exit(app.exec_())
