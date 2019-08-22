#!/usr/bin/python
from Graph import Coordinate

class FileHandler():
    @staticmethod
    def read_coordinates(coordinate_file_path):
        coordinates = list([])
        coordinate_index = 0

        print(coordinate_file_path)

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
