#!/usr/bin/python
from Graph import Vertex
import numpy as np
import pandas as pd
import re
import os
import time

class FileHandler():
    @staticmethod
    def read_adjacency_matrix(adjacency_matrix_file_path):
        adjacency_matrix = pd.read_csv(adjacency_matrix_file_path, dtype='int')
        return adjacency_matrix

    @staticmethod
    def read_graph(vertex_file_path, adjacency_matrix=None):
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
                    vertices = np.concatenate((vertices, Vertex(int(index), float(x), float(y))), axis=None)
                    # Increment the vertex index by 1
                    vertex_index += 1
        # When finished, close the file.
        finally:
            vertex_file.close()

        print(adjacency_matrix)
        print(vertices)

        if adjacency_matrix is None:
            # Create a list of adjacent vertices for all vertices in the list
            for index, vertex in enumerate(vertices):
                vertex.adjacent_vertices = [adjacent_vertex for adjacent_vertex in vertices if adjacent_vertex != vertex]
        else:
            for row_index, row in adjacency_matrix.iterrows():
                for column_index, relation in row.items():
                    if relation == 1:
                        vertices[row_index].adjacent_vertices.append(vertices[int(column_index)-1])

        # Return the list of Vertex objects
        return vertices

    @staticmethod
    def enforce_path(purposed_path):
        # If folders don't exist along the path, create them.
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

        # Open file
        with open(route_log_path, "r") as route_log:
            # Read the first line from the route_log
            route_line = route_log.readline()

            # Continue to read lines until EOF
            while route_line != '':
                # Initialize minimum_route_distance to first route in file if not doneso.
                if minimum_route_distance is None:
                    data = route_line.split('|')
                    # Retrieve the list from string representation.
                    minimum_route_list = [int(re.search(r'\d+', vertex).group()) for vertex in data[0].split(',')]
                    # Retrieve distance traveled
                    minimum_route_distance = float(data[1])
                else:
                    data = route_line.split('|')
                    # Retrieve the list from string representation.
                    route_list = [int(re.search(r'\d+', vertex).group()) for vertex in data[0].split(',')]
                    # Retrieve distance traveled
                    route_distance = float(data[1])

                    # Update new minimum if shorter distance.
                    if route_distance < minimum_route_distance:
                        minimum_route_list = route_list
                        minimum_route_distance = route_distance

                    route_line = route_log.readline()

        # return the identified minimum route vertex order and distance traveled.
        return minimum_route_list, minimum_route_distance
