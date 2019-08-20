#!/usr/bin/python

import matplotlib.pyplot as plt

class Route(object):
    def __init__(self, starting_spot):
        self.city_order = list([0])
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

class Coordinate(object):
    def __init__(self, coordinate_id, x, y):
        self.coordinate_id = coordinate_id
        self.x = x
        self.y = y
        self.distances = list([])
        self.visited = False

    def __eq__(self, other):
        self.x = other.x
        self.y = other.y
        self.distances = other.distances
        self.visited = other.visited

    def compute_distance(self, other_coordinate):
        return ((self.x - other_coordinate.x)**2 + (self.y - other_coordinate.y)**2)**0.5

    def __str__(self):
        return "(ID: " + str(self.coordinate_id) + ", X: " + str(self.x) + ", Y: " + str(self.visited) + ")"

class CoordinateSet(object):
    def __init__(self, coordinate_file_path):
        self.coordinates = CoordinateSet.read_coordinates(coordinate_file_path)
        self.distance_graph = list([])

    def build_graph(self):
        for coordinate1 in self.coordinates:
            for coordinate2 in self.coordinates:
                coordinate2.distances.append(coordinate2.compute_distance(coordinate1))
            self.distance_graph.append(coordinate1.distances)

    def display_graph(self):
        for coordinate in self.coordinates:
            print(coordinate)

        print("")

        for distances in self.distance_graph:
            print(distances)

    def get_updated_distance_list_for_coordinate(self, current_coordinate):
        updated_distance_list = current_coordinate.distances
        
        for coordinate_index, coordinate in enumerate(self.coordinates):
            if coordinate.visited:
                updated_distance_list[coordinate_index] = 0.0

        return updated_distance_list

    def greedy_tsm_solution(self, distance_traveled=0):
        route = Route(0)

        min_distance = 0
        min_distance_index = 0
        current_coordinate = self.coordinates[0]
        self.coordinates[0].visited = True
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

    def brute_force_tsm_solution(self, distance_traveled = 0):
        route = Route(0)
        print(self.distance_graph[0])
        return route

    def try_all_open_routes_from_node():
        pass

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

        return coordinates

    def plot_coordinates(self):
        x = list([])
        y = list([])
        
        for coordinate in self.coordinates:
            x.append(coordinate.x)
            y.append(coordinate.y)

        plt.plot(x,y)
    

if __name__ == "__main__":
    coordinate_file_paths = list([])

    for i in range(4, 13-8):
        coordinate_file_paths.append("/Users/sonia66/Downloads/Project1/Random{}.tsp".format(i))

    for coordinate_file_path in coordinate_file_paths:
        print("\n\n\t==== Testing coordinate_file_path: " + coordinate_file_path + " ====")
        coordinate_set = CoordinateSet(coordinate_file_path)
        coordinate_set.build_graph()

        print("\n=== Displaying Graph ===")

        coordinate_set.display_graph()
        coordinate_set.plot_coordinates()

        print("\n=== Displaying Solution ===")

        print(coordinate_set.brute_force_tsm_solution())
