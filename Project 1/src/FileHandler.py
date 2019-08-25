#!/usr/bin/python
from Graph import Vertex
import numpy as np
import re
import os
import time

class FileHandler():
    @staticmethod
    def read_vertices(vertex_file_path):
        # Initialize list and index
        vertices = np.array([])
        vertex_index = 0

        # Try to open the file
        try:
            vertex_file = open(vertex_file_path, "r")

            # If successful, read each line of the file
            for line in vertex_file.readlines():
                # once a first character digit is found, we've reached the data section
                if line[0].isdigit():
                    # split the line into index, x, and y values
                    index, x, y = line.split(" ")
                    # Create a vertex object out of these attributes and append it to the list of vertices
                    vertices = np.concatenate((vertices, Vertex(vertex_index, float(x), float(y))), axis = None)
                    # Increment the vertex index by 1
                    vertex_index += 1
        # When finished, close the file.
        finally:
            vertex_file.close()

        # Create a list of adjacent vertices for all vertices in the list
        for index, vertex in enumerate(vertices):
            vertex.adjacent_vertices = [adjacent_vertex for adjacent_vertex in vertices if adjacent_vertex != vertex]

        # Return the list of Vertex objects
        return vertices

    @staticmethod
    def enforce_path(purposed_path):
        if not os.path.exists(purposed_path):
            os.makedirs(purposed_path)

    @staticmethod
    def log_route(route, route_log_path):
        route_log = open(route_log_path, "a+")

        route_log.write(str(route) + "\n")

        route_log.close()

    @staticmethod
    def find_minimum_route(route_log_path):
        minimum_route_list = list([])
        minimum_route_distance = None

        with open(route_log_path, "r") as route_log:
            route_line = route_log.readline()

            while(route_line != ''):
                if minimum_route_distance is None:
                    data = route_line.split('|')
                    minimum_route_list = [int(re.search(r'\d+', vertex).group()) for vertex in data[0].split(',')]
                    minimum_route_distance = float(data[1])

                else:
                    data = route_line.split('|')
                    route_list = [int(re.search(r'\d+', vertex).group()) for vertex in data[0].split(',')]
                    route_distance = float(data[1])

                    if route_distance < minimum_route_distance:
                        minimum_route_list = route_list
                        minimum_route_distance = route_distance

                route_line = route_log.readline()

        return minimum_route_list, minimum_route_distance
