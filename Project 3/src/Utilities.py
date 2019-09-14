import enum

class Math():
    @staticmethod
    def calculate_distance_from_line_to_point(line, point):
        line_point_1, line_point_2 = line
        x1, y1 = line_point_1
        x2, y2 = line_point_2
        x0, y0 = point
        return abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / ((x2-x1)**2 + (y2-y1)**2)**0.5

    @staticmethod
    def lines_intersect(line1, line2):
        line1_point1, line1_point2 = line1
        line2_point1, line2_point2 = line2

        d1 = Math.get_line_point_direction(line1, line2_point1)
        d2 = Math.get_line_point_direction(line1, line2_point2)
        d3 = Math.get_line_point_direction(line2, line1_point1)
        d4 = Math.get_line_point_direction(line2, line1_point2)

        # Points are intersecting
        if d1 != d2 and d3 != d4:
            return True

        if d1 == Math.LINE_POINT_DIRECTIONS.COLINEAR and Math.is_point_on_line(line1, line2_point1):
            return True

        if d2 == Math.LINE_POINT_DIRECTIONS.COLINEAR and Math.is_point_on_line(line1, line2_point2):
            return True

        if d3 == Math.LINE_POINT_DIRECTIONS.COLINEAR and Math.is_point_on_line(line2, line1_point1):
            return True

        if d4 == Math.LINE_POINT_DIRECTIONS.COLINEAR and Math.is_point_on_line(line2, line1_point2):
            return True

        return False

    @staticmethod
    def is_point_on_line(line, point):
        line_point_1, line_point_2 = line
        x1, y1 = line_point_1
        x2, y2 = line_point_2
        x0, y0 = point

        if x0 <= max([x1, x2]) and x0 >= min([x1, x2]) and y0 <= max([y1, y2]) and y0 >= min([y1, y2]):
            return True

        return False

    @staticmethod
    def get_line_point_direction(line, point):
        line_point_1, line_point_2 = line
        x1, y1 = line_point_1
        x2, y2 = line_point_2
        x0, y0 = point

        direction_calculation = (y2-y1)*(x0-x2)-(x2-x1)*(y0-y2)

        if direction_calculation == 0:
            return Math.LINE_POINT_DIRECTIONS.COLINEAR
        elif direction_calculation < 0:
            return Math.LINE_POINT_DIRECTIONS.CCW
        else:
            return Math.LINE_POINT_DIRECTIONS.CW

    class LINE_POINT_DIRECTIONS(enum.IntEnum):
        COLINEAR = enum.auto()
        CCW = enum.auto()
        CW = enum.auto()

    @staticmethod
    def color_quantization(value, number_of_bins):
        assert 1 <= value <= number_of_bins, "Invalid vertex_id for color_quantization ({}) for color look_up.  " \
                                     "Please update date to fit new range.".format(vertex_id)

        size_of_bins = (256 - 0 + 1) / number_of_bins

        color_value = hex(int(size_of_bins) * value)[2:]

        if int(color_value, 16) < 10:
            return "0" + color_value
        else:
            return color_value

if __name__ == "__main__":
    line_point_1 = (-1, 1)
    line_point_2 = (1, 1)
    line = (line_point_1, line_point_2)
    point = (0, 0)

    print(Math.calculate_distance_from_line_to_point(line, point))
    print(Math.is_point_on_line(line, point))
    point = (-0.5, 1)
    print(Math.is_point_on_line(line, point))
