import sys
import os

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random

from FileHandler import FileHandler
from Graph import Graph

class TravelingSalesmanGUI(QMainWindow):
    def __init__(self, graph=None, algorithm=None):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Traveling Salesman Problem'
        self.width = 940
        self.height = 640
        self.graph = graph
        self.algorithm = algorithm
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        m = PlotCanvas(self, width=8, height=6.4)
        m.move(0,0)

        button = QPushButton('Step Forward', self)
        button.setToolTip('Increase number of cities')
        button.move(800,0)
        button.resize(140,100)

        button = QPushButton('Step Backward', self)
        button.setToolTip('Decrease number of cities')
        button.move(800,100)
        button.resize(140,100)

        button = QPushButton('Run Simulation', self)
        button.setToolTip('Run simultion until completion.')
        button.move(800,200)
        button.resize(140,100)

        button = QPushButton('Show All Edges', self)
        button.setToolTip('Run simultion until completion.')
        button.move(800,300)
        button.resize(140,100)

        self.show()


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
                                                              str(hex(int(random.uniform(0.2, 1)*256)))[2:],
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
        ex = TravelingSalesmanGUI(graph)
        sys.exit(app.exec_())
