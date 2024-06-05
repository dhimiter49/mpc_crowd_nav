import numpy as np


def gen_polygon(radius, sides=8):
    def rot_mat(rad):
        return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    polygon = [[radius, 0]]
    for i in range(1, sides + 1):
        polygon.append(rot_mat(2 * np.pi / sides) @ polygon[i - 1])
    polygon_lines = []
    for i in range(sides):
        m = (polygon[i][1] - polygon[i + 1][1]) / (polygon[i][0] - polygon[i + 1][0])
        b = polygon[i][1] - m * polygon[i][0]
        polygon_lines.append([m, b])
    return np.array(polygon_lines)
