import numpy as np
import pandas as pd
import math
import operator
import random
import matplotlib.pyplot as plt

class TSP_ACO(object):
    def __init__(self):
        """
        TSP object, to be solved using ACO

        metric:
        """

        # Initialize
        self.pheromones = []

        # Initialize the pheromones for each edge


class ACO(object):
    def __init__(self, max_generations:int, num_ants:int, prob, rho:float, Q:float):
        """
        ACO Problem object

        max_generations: maximum number of generations
        num_ants: number of ants to generate
        prob: (lambda) function for assigning the probability of ant k choosing the edge ij
            i.e. prob(k, i, j) -> [0, 1]
        rho: pheromone evaporation coefficient
        Q: constant for deter
        """
        # Parameter initializations
        self.max_generations = max_generations
        self.num_ants = num_ants
        self.prob = prob

        self.rho = rho
        self.Q = Q

        # Results
        self.optimal_cost = math.inf
        self.optimal_path = []
    #
    # def solve(self):
    #     for gen in range(self.max_generations):


# class Ant(object):
    # def __init__(self):