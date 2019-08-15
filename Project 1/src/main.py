#!/usr/bin/python

import matplotlib.pyplot as plt

class Coordinate(object):
    def __init__(self, x, y):
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

    def display(self):
        print("(X: " + str(self.x) + ", Y: " + str(self.y) + ", V: " + str(self.visited) + ")")

    @staticmethod
    def copy(source_coord, dest_coord):
        dest_coord.x = source_coord.x
        dest_coord.y = source_coord.y
        dest_coord.distances = source_coord.distances
        dest_coord.visisted = source_coord.visited

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
            coordinate.display()

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
        city_order = list([0])

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

                city_order.append(min_distance_index)
                distance_traveled += min_distance

        distance_traveled += self.distance_graph[0][city_order[len(city_order)-1]]
        city_order.append(0)

        return city_order, distance_traveled

    @staticmethod
    def read_coordinates(cordinate_file_path):
        coordinates = list([])
        
        try:
            coordinate_file = open(coordinate_file_path, "r")

            for line in coordinate_file.readlines():
                if line[0].isdigit():
                    index, x, y = line.split(" ")
                    coordinates.append(Coordinate(float(x), float(y)))
                                       
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

    for i in range(4, 13):
        coordinate_file_paths.append("/Users/sonia66/Downloads/Project1/Random{}.tsp".format(i))

    for coordinate_file_path in coordinate_file_paths:
        print("\n\n\t==== Testing coordinate_file_path: " + coordinate_file_path + " ====")
        coordinate_set = CoordinateSet(coordinate_file_path)
        coordinate_set.build_graph()

        print("\n=== Displaying Graph ===")

        coordinate_set.display_graph()
        coordinate_set.plot_coordinates()

        print("\n=== Displaying Solution ===")

        print(coordinate_set.greedy_tsm_solution())
