#!/usr/bin/python
import matplotlib.pyplot as plt

class Route(object):
    def __init__(self, current_city_order, coordinate_graph):
        self.city_order = current_city_order
        self.coordinate_graph = coordinate_graph
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
        return str(self.city_order) + ", " + str(self.distance_traveled)

    def goto(self, coordinate_id):
        if len(self.city_order) == 0:
            self.distance_traveled = 0
            self.city_order.append(coordinate_id)
            self.coordinate_graph.coordinates[coordinate_id].visited = True
        else:
            self.distance_traveled += self.coordinate_graph.coordinates[self.city_order[-1]].compute_distance(self.coordinate_graph.coordinates[coordinate_id])
            self.city_order.append(coordinate_id)
            self.coordinate_graph.coordinates[coordinate_id].visited = True

class Coordinate(object):
    def __init__(self, coordinate_id, x, y, visited=False):
        self.coordinate_id = coordinate_id
        self.x = x
        self.y = y
        self.adjacent_coordinates = list([])
        self.distances = list([])
        self.visited = visited

    def __eq__(self, other):
        return self.coordinate_id == other.coordinate_id

    def __str__(self):
        result = "(ID: " + str(self.coordinate_id) + ", X: " + str(self.x) + ", Y: " + str(self.y) + ", V:" + str(self.visited) + ")\n"

        return result[:-1]

    def compute_distance(self, other_coordinate):
        return ((self.x - other_coordinate.x)**2 + (self.y - other_coordinate.y)**2)**0.5

    def display(self):
        print(self)
        for coordinate in self.adjacent_coordinates:
            print(coordinate)

    def copy(self, other):
        self.coordinate_id = other.coordinate_id
        self.x = other.x
        self.y = other.y

class Graph(object):
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.distance_graph = list([])

    def __str__(self):
        string = ""

        for coordinate in self.coordinates:
            string += str(coordinate)
            string += "\n"

        string += "\n"


        for distances in self.distance_graph:
            string += str(distances)
            string += '\n'

        return string

    def build_graph(self):
        for coordinate1 in self.coordinates:
            for coordinate2 in self.coordinates:
                coordinate2.distances.append(coordinate2.compute_distance(coordinate1))
            self.distance_graph.append(coordinate1.distances)

        return self.distance_graph

    def plot_coordinates(self):
        x = list([])
        y = list([])

        for coordinate in self.coordinates:
            x.append(coordinate.x)
            y.append(coordinate.y)

        plt.plot(x,y)

    def get_unvisited_coordinate_ids(self):
        return [coordinate.coordinate_id for coordinate in self.coordinates if not coordinate.visited]

    def finished(self):
        if False in [coordinate.visited for coordinate in self.coordinates]:
            return False
        else:
            return True
