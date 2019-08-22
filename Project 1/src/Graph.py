#!/usr/bin/python
import matplotlib.pyplot as plt

class Route(object):
    def __init__(self, current_vertex_order, graph):
        self.vertex_order = current_vertex_order
        self.graph = graph
        self.distance_traveled = 0

    def __eq__(self, other):
        return self.distance_traveled == other.distance_traveled

    def __lt__(self, other):
        return self.distance_traveled < other.distance_traveled

    def __le__(self, other):
        return self.distance_traveled <= other.distance_traveled

    def __gt__(self, other):
        return self.distance_traveled > other.distance_traveled

    def __ge__(self, other):
        return self.distance_traveled >= other.distance_traveled

    def __str__(self):
        return str(self.vertex_order) + ", " + str(self.distance_traveled)

    def goto(self, vertex_id):
        if len(self.vertex_order) == 0:
            self.distance_traveled = 0
            self.vertex_order.append(vertex_id)
            self.graph.vertices[vertex_id].visited = True
        else:
            self.distance_traveled += self.graph.vertices[self.vertex_order[-1]].compute_distance(self.graph.vertices[vertex_id])
            self.vertex_order.append(vertex_id)
            self.graph.vertices[vertex_id].visited = True

class Vertex(object):
    def __init__(self, vertex_id, x, y, visited=False):
        self.vertex_id = vertex_id
        self.x = x
        self.y = y
        self.adjacent_vertices = list([])
        self.distances = list([])
        self.visited = visited

    def __eq__(self, other):
        return self.vertex_id == other.vertex_id

    def __str__(self):
        result = "(ID: " + str(self.vertex_id) + ", X: " + str(self.x) + ", Y: " + str(self.y) + ", V:" + str(self.visited) + ")\n"

        return result[:-1]

    def compute_distance(self, other_vertex):
        return ((self.x - other_vertex.x)**2 + (self.y - other_vertex.y)**2)**0.5

    def display(self):
        print(self)
        for vertex in self.adjacent_vertices:
            print(vertex)

    def copy(self, other):
        self.vertex_id = other.vertex_id
        self.x = other.x
        self.y = other.y

class Graph(object):
    def __init__(self, vertices):
        self.vertices = vertices
        self.edges = list([])

    def __str__(self):
        string = ""

        for vertex in self.vertices:
            string += str(vertex)
            string += "\n"

        string += "\n"


        for edge in self.edges:
            string += str(edge)
            string += '\n'

        return string

    def build_graph(self):
        for vertex1 in self.vertices:
            for vertex2 in self.vertices:
                vertex2.distances.append(vertex2.compute_distance(vertex1))
            self.edges.append(vertex1.distances)

        return self.edges

    def plot_graph(self):
        x = list([])
        y = list([])

        for vertex in self.vertices:
            x.append(vertex.x)
            y.append(vertex.y)

        plt.plot(x,y)

    def get_unvisited_vertex_ids(self):
        return [vertex.vertex_id for vertex in self.vertices if not vertex.visited]

    def finished(self):
        if False in [vertex.visited for vertex in self.vertices]:
            return False
        else:
            return True
