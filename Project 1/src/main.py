#!/usr/bin/python
from copy import deepcopy

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

class CoordinateGraph(object):
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

class FileHandler():
    @staticmethod
    def read_coordinates(cordinate_file_path):
        coordinates = list([])
        coordinate_index = 0

        try:
            coordinate_file = open(coordinate_file_path, "r")

            for line in coordinate_file.readlines():
                if line[0].isdigit():
                    index, x, y = line.split(" ")
                    coordinates.append(Coordinate(coordinate_index, float(x), float(y)))
                    coordinate_index += 1
        finally:
            coordinate_file.close()

        for index, coordinate in enumerate(coordinates):
            coordinate.adjacent_coordinates = [adjacent_coordinate for adjacent_coordinate in coordinates if adjacent_coordinate != coordinate]

        return coordinates

class TravelingSalesman():
    @staticmethod
    def brute_force_solution(coordinate_graph, current_coordinate_id=0, distance_traveled = 0):
        # Recursive function for trying all adjacent nodes.
        def try_all_open_routes_from_current_route(route):
            routes = list([])
            current_coordinate = route.coordinate_graph.coordinates[current_coordinate_id]

            new_route = list([])

            for adjacent_coordinate in [adjacent_coordinate for adjacent_coordinate in current_coordinate.adjacent_coordinates if adjacent_coordinate.visited == False]:
                new_route = deepcopy(route)
                new_route.goto(adjacent_coordinate.coordinate_id)

                if(new_route.coordinate_graph.finished()):
                    new_route.goto(current_coordinate_id)
                    routes.append(new_route)
                else:
                    routes.extend(try_all_open_routes_from_current_route(new_route))

            return routes

        # Initialize the route and go to the starting point
        route = Route([], coordinate_graph)
        route.goto(current_coordinate_id)
        routes = list([])

        # Recursively try all open routes from the current route, advancing when possible.
        routes.extend(try_all_open_routes_from_current_route(route))

        # Identify the route with minimum distance traveled
        minimum_route = deepcopy(routes[0])

        for route in routes:
            if route < minimum_route:
                minimum_route = deepcopy(route)

        # return minimum route
        return minimum_route

'''
    @staticmethod
    def greedy_solution(coordinate_graph, current_coordinate_id=0, distance_traveled=0):
        route = Route(current_coordinate_id, coordinate_graph)

        min_distance = 0
        min_distance_index = 0

        unvisited_coordinates = coordinate_graph.get_unvisited_coordinate_ids()

        while(len(unvisited_coordinates) != 0):
            # Get distances from current coordinate
            coordinate_distances = [distance for distance in coordinate_graph.coordinates[current_coordinate_id].distances if distance != 0.0]
            min_distance = min(coordinate_distances)

            print("min_distance", min_distance)

            # Determine coordinate of minimum distance
            for index, distance in enumerate(coordinate_graph.coordinates[current_coordinate_id].distances):
                if distance <= min_distance and distance != 0:
                    min_distance = distance
                    min_distance_index = index
                    closest_coordinate = coordinate_graph.coordinates[min_distance_index]

            print("closest_coordinate.coordinate_id:", closest_coordinate.coordinate_id)
            print("current_coordinate_id", current_coordinate_id)

            print("distance_traveled from " + " = " + str(coordinate_graph.distance_graph[current_coordinate_id][closest_coordinate.coordinate_id]))
            print("distance_traveled from " + " = " + str(coordinate_graph.distance_graph[closest_coordinate.coordinate_id][current_coordinate_id]))

            current_coordinate_id = closest_coordinate.coordinate_id
            closest_coordinate.visited = True
            unvisited_coordinates = coordinate_graph.get_unvisited_coordinate_ids()

        for coordinate in coordinate_graph.get_unvisited_coordinate_ids():
            print(coordinate)

'''
'''
        min_distance = 0
        min_distance_index = 0
        current_coordinate = coordinate_graph.coordinates[0]
        coordinate_graph.coordinates[0].visited = True
        distance_list = self.get_updated_distance_list_for_coordinate(current_coordinate)

            while(True):
            min_distance = max(distance_list)
            min_distance_index = distance_list.index(min_distance)

            if min_distance == 0.0:
                break
            else:
                for index, distance in enumerate(distance_list):
                    if distance <= min_distance and distance != 0.0:
                        min_distance = distance
                        min_distance_index = index

                current_coordinate = self.coordinates[min_distance_index]
                self.coordinates[min_distance_index].visited = True
                distance_list = self.get_updated_distance_list_for_coordinate(current_coordinate)

                route.city_order.append(min_distance_index)
                route.distance_traveled += min_distance

        route.distance_traveled += self.distance_graph[0][route.city_order[len(route.city_order)-1]]
        route.city_order.append(0)

        return route
'''


if __name__ == "__main__":
    coordinate_file_paths = list([])

    for i in range(4, 13):
        coordinate_file_paths.append("/Users/sonia66/Downloads/Project1/Random{}.tsp".format(i))

    for coordinate_file_path in coordinate_file_paths:
        print("\n\n\t==== Testing coordinate_file_path: " + coordinate_file_path + " ====")
        coordinates = FileHandler.read_coordinates(coordinate_file_path)

        coordinate_graph = CoordinateGraph(coordinates)
        coordinate_graph.build_graph()

        print("\n=== Displaying Graph ===")

        print(coordinate_graph)

        print("\n=== Displaying Solution ===")

        print("brute_force_solution", str(TravelingSalesman.brute_force_solution(coordinate_graph)))
