#!/usr/bin/python
import sys
import os
from copy import deepcopy
from FileHandler import FileHandler
from Graph import Route, Graph
import numpy as np


class TravelingSalesman():
    @staticmethod
    def brute_force_solution(graph, current_vertex_id=0, distance_traveled = 0):
        # Recursive function for trying all adjacent vertices.
        def try_all_open_routes_from_current_route(route):
            # Initialize Routes to keep track of all attempted routes.
            routes = np.array([])
            # Start at the current vertex id location
            current_vertex = route.graph.vertices[current_vertex_id]

            # For each adjacent vertex that has not been visited
            for adjacent_vertex in current_vertex.get_unvisited_adjacent_vertex_ids():
                # copy the route so far
                new_route = deepcopy(route)
                # goto the current adjacent_vertex
                new_route.goto(adjacent_vertex.vertex_id)

                # if all vertices have been visisted
                if(new_route.graph.finished()):
                    # goto the starting point
                    new_route.goto(current_vertex_id)
                    # append the route to the list of completed routes
                    routes = np.concatenate((routes, new_route), axis=None)
                else: # if not,
                    # Recall the recursive function using the updated route.
                    routes = np.concatenate((routes, try_all_open_routes_from_current_route(new_route)), axis=None)

            # After all adjacent vertices have been visisted recursively, return the list of routes
            return routes

        # Initialize the route
        route = Route(list([]), graph)
        # goto the current vertex id
        route.goto(current_vertex_id)

        # Initialize a list of routes
        routes = np.array([])

        # Recursively try all open routes from the current route, advancing when possible.
        routes = np.concatenate((routes, try_all_open_routes_from_current_route(route)), axis=None)

        # Identify the route with minimum distance traveled
        return min(routes)

if __name__ == "__main__":
    # Retrieve command line arguments
    args = sys.argv

    if len(args) != 3:
        print("Command Line Arguments should follow the format:")
        print("TrainingSalesman.py [solve_method] [relative path to vertex_graph_file]")
        print("\nImplemented solve_methods include: brute_force")
    else:
        # retrieve solve_method
        solve_method = sys.argv[1]
        # retrieve relative path to vertex_graph_file
        vertex_graph_file = sys.argv[2]

        # Read the vertices from the vertex graph file.
        vertices = FileHandler.read_vertices(os.getcwd() + os.sep + vertex_graph_file)

        # Build a graph representing the vertices and edges.
        graph = Graph(vertices)
        # Calculate edges
        graph.build_graph()

        # Display the graph before solving.  TODO: Replace with plotting
        print("\n=== Displaying Graph ===")
        print(graph)
        graph.plot()

        # Solve the graph using the solve_method provided
        print("\n=== Displaying Solution ===")
        if solve_method == 'brute_force':
            result = TravelingSalesman.brute_force_solution(graph)
            print("brute_force_solution", str(result))
            result.plot()
        else:
            print("Invalid solve_method.  Current implemented solve methods include: brute_force")
