#!/usr/bin/python
import sys
import os
from copy import deepcopy
from FileHandler import FileHandler
from Graph import Route, Graph

class TravelingSalesman():
    @staticmethod
    def brute_force_solution(graph, current_vertex_id=0, distance_traveled = 0):
        # Recursive function for trying all adjacent vertices.
        def try_all_open_routes_from_current_route(route):
            routes = list([])
            new_route = list([])
            current_vertex = route.graph.vertices[current_vertex_id]

            for adjacent_vertex in [adjacent_vertex for adjacent_vertex in current_vertex.adjacent_vertices if adjacent_vertex.visited == False]:
                new_route = deepcopy(route)
                new_route.goto(adjacent_vertex.vertex_id)

                if(new_route.graph.finished()):
                    new_route.goto(current_vertex_id)
                    routes.append(new_route)
                else:
                    routes.extend(try_all_open_routes_from_current_route(new_route))

            return routes

        # Initialize the route and go to the starting point
        route = Route([], graph)
        route.goto(current_vertex_id)
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

if __name__ == "__main__":
    args = sys.argv

    solve_method = sys.argv[1]
    content_file = sys.argv[2]

    vertices = FileHandler.read_vertices(os.getcwd() + os.sep + content_file)

    graph = Graph(vertices)
    graph.build_graph()

    print("\n=== Displaying Graph ===")

    print(graph)

    print("\n=== Displaying Solution ===")

    if solve_method == 'brute_force':
        print("brute_force_solution", str(TravelingSalesman.brute_force_solution(graph)))
