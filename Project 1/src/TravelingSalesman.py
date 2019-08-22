#!/usr/bin/python
import sys
import os
from copy import deepcopy
from FileHandler import FileHandler
from Graph import Route, Graph

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
    args = sys.argv

    content_file = sys.argv[1]
    print("\n\n\t==== Testing coordinate_file_path: " + content_file + " ====")
    coordinates = FileHandler.read_coordinates(os.getcwd() + os.sep + content_file)

    coordinate_graph = Graph(coordinates)
    coordinate_graph.build_graph()

    print("\n=== Displaying Graph ===")

    print(coordinate_graph)

    print("\n=== Displaying Solution ===")

    print("brute_force_solution", str(TravelingSalesman.brute_force_solution(coordinate_graph)))
