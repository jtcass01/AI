#!/usr/bin/python
from Graph import Vertex

class FileHandler():
    @staticmethod
    def read_vertices(vertex_file_path):
        vertices = list([])
        vertex_index = 0

        try:
            vertex_file = open(vertex_file_path, "r")

            for line in vertex_file.readlines():
                if line[0].isdigit():
                    index, x, y = line.split(" ")
                    vertices.append(Vertex(vertex_index, float(x), float(y)))
                    vertex_index += 1
        finally:
            vertex_file.close()

        for index, vertex in enumerate(vertices):
            vertex.adjacent_vertices = [adjacent_vertex for adjacent_vertex in vertices if adjacent_vertex != vertex]

        return vertices
