from point import Point
from numpy import mean, var
import math


class DummyNormalizer:
    def fit(self, points):
        pass

    def transform(self, points):
        return points

    def print_name(self):
        return "DummyNormalizer"

class ZNormalizer:
    def __init__(self):
        self.mean_variance_list = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.mean_variance_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.mean_variance_list.append([mean(values), var(values, ddof=1)**0.5])

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.mean_variance_list[i][0]) / self.mean_variance_list[i][1]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new

    def print_name(self):
        return "ZNormalizer"


class SumNormalizer:
    def __init__(self):
        self.abs_sum = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.abs_sum = []
        for i in range(len(all_coordinates[0])):
            values = [abs(x[i]) for x in all_coordinates]
            self.abs_sum.append(sum(values))

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i]) / self.abs_sum[i]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new

    def print_name(self):
        return "SumNormalizer"


class MinMaxNormalizer:
    def __init__(self):
        self.min_max = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.min_max = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.min_max.append([max(values), min(values)])

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [((new_coordinates[i]) - self.min_max[i][1]) / (self.min_max[i][0] - self.min_max[i][1])
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new

    def print_name(self):
        return "MinMaxNormalizer"