import enum
import numpy as np

class Math():
    @staticmethod
    def calculate_distance_from_line_to_point(line, point):
        lp1, lp2 = line
        x1, y1 = lp1
        x2, y2 = lp2
        x0, y0 = point

        lp1_to_p = Math.calculate_distance_from_point_to_point(lp1, point)
        lp2_to_p = Math.calculate_distance_from_point_to_point(lp2, point)
        a = x0 - x1
        b = y0 - y1
        c = x2 - x1
        d = y2 - y1

        dot = a*c + b * d
        len_sq = c*c + d*d
        param = -1

        if len_sq != 0:
            param = dot / len_sq

        if param < 0:
            xx = x1
            yy = y1
        elif param > 1:
            xx = x2
            yy = y2
        else:
            xx = x1 + param*c
            yy = y1 + param*d

        dx = x0 - xx
        dy = y0 - yy

        result = (dx**2 + dy**2)**0.5

        if (x0 < min(x1, x2) or x0 > max(x1,x2)) and (y0 < min(y1, y2) or y0 > max(y1, y2)):
            lp1_to_p = Math.calculate_distance_from_point_to_point(lp1, point)
            lp2_to_p = Math.calculate_distance_from_point_to_point(lp2, point)

            if lp1_to_p < lp2_to_p:
                return lp1_to_p
            else:
                return lp2_to_p
        else:
            a = x0 - x1
            b = y0 - y1
            c = x2 - x1
            d = y2 - y1

            dot = a*c + b * d
            len_sq = c*c + d*d
            param = -1

            if len_sq != 0:
                param = dot / len_sq

            if param < 0:
                xx = x1
                yy = y1
            elif param > 1:
                xx = x2
                yy = y2
            else:
                xx = x1 + param*c
                yy = y1 + param*d

            dx = x0 - xx
            dy = y0 - yy

            result = (dx**2 + dy**2)**0.5

            return result

    @staticmethod
    def calculate_distance_from_point_to_point(point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    @staticmethod
    def lines_intersect(line1, line2):
        line1_point1, line1_point2 = line1
        line2_point1, line2_point2 = line2

        if (line1_point1 == line2_point1).any() or (line1_point1 == line2_point2).any() or (line1_point2 == line2_point1).any() or (line1_point2 == line2_point2).any():
            return False

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
                                     "Please update date to fit new range.".format(value)

        size_of_bins = (256 - 0 + 1) / number_of_bins

        color_value = hex(int(size_of_bins) * value)[2:]

        if len(color_value) == 1:
            return "0" + color_value
        else:
            return color_value

    @staticmethod
    def normalize_rgb(value, min_possible_value, max_possible_value):
        output = int(255 - ((value - max_possible_value) * 255 / (min_possible_value - max_possible_value)))

        if output < 0:
            return "00"
        elif output > 255:
            return "FF"
        else:
            output = hex(output)[2:]

            if len(output) == 1:
                return "0" + output
            else:
                return output


if __name__ == "__main__":
    line_point_1 = (-1, 1)
    line_point_2 = (1, 1)
    line = (line_point_1, line_point_2)
    point = (0, 0)

    print(Math.calculate_distance_from_line_to_point(line, point))
    print(Math.is_point_on_line(line, point))
    point = (-0.5, 1)
    print(Math.is_point_on_line(line, point))
