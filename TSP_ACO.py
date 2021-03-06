"""
Created on 2020/10/02
@author: nicklee

(Description)
"""
from collections import defaultdict
# from tuning_PSO import *
import math
import random
import numpy as np
from Errors import *


class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("Cannot modify ReadOnlyDict")
    __setitem__ = __readonly__
    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__


class TSP_ACO(object):
    def __init__(self, cities, cities_nums, metric):
        """
        TSP object, to be solved using ACO

        Attributes:
            cities -- numpy array of 2D-coordinates of cities (N x 2)
                i.e. np.array([[x_0, y_0], [x_1, y_1], ..., [x_{N-1}, y_{N-1}]])
            cities_nums -- numpy array of numbers(names) of cities (N x 1)
                i.e. np.array([n_0, n_1, ..., n_{N-1}])
            metric -- (lambda) function for assigning weight at each edge ij
                i.e. metric(i, j) -> [0, inf)
            distances -- defaultdict of distances of all the edges
                i.e. distances((i, j)) -> [0, inf)
            pheromones -- dict containing the pheromone level of each edge ij
                i.e. pheromones[(i, j)] -> [0, inf)
        """
        # Initialize
        self.cities = cities
        self.N = len(cities)
        self.cities_nums = cities_nums

        self.metric = metric
        self.distances = dict()
        self.pheromones = dict()
        for i in range(self.N):
            self.distances[(i, i)] = 0
            self.pheromones[(i, i)] = 0
            for j in range(i):
                # Calculate the distances
                self.distances[(i, j)] = self.metric(self.cities[i], self.cities[j])
                self.distances[(j, i)] = self.metric(self.cities[i], self.cities[j])
                # Initialize the pheromones
                self.pheromones[(i, j)] = 1 / (self.N * 10)
                self.pheromones[(j, i)] = 1 / (self.N * 10)

        self.distances = ReadOnlyDict(self.distances)   # Prevent modifications!

    def initialize_Q(self):
        """
        Initialize Q to a length of some (pseudo-)randomly-chosen Hamiltonian cycle
        :return: Q
        """
        Q = 0
        for i in range(self.N - 1):
            Q += self.distances[(i, i + 1)]
        Q += self.distances[(self.N - 1, 0)]
        return Q

    def length_path(self, vertices, cycle=True):
        """
        For a given sequence of vertices, calculate the total length of the path.
        If cycle=True, then output the total length when it is regarded as a cycle

        :param vertices: List of vertices
        :param cycle: Boolean indicating whether the given path is a cycle

        :return: total length of the path
        """
        V = len(vertices)
        total_len = 0
        for i in range(V - 1):
            total_len += self.distances[(vertices[i], vertices[i + 1])]
        if cycle:
            total_len += self.distances[(vertices[V - 1], 0)]
        return total_len

    def optimize(self, T, num_ants, a, b, rho, Q):
        """
        Wrapper function for ACO.optimize() using the given parameters

        :param T: maximum number of rounds (or maximum amount of time)
        :param num_ants: number of ants to generate
        :param a: exponent determining the dependency on the amount of pheromone deposit
        :param b: exponent determining the dependency on the distance
        :param rho: pheromone evaporation coefficient (a real number between 0 and 1)
        :param Q: constant for deter

        :return: optimal_cost, optimal_path
        """
        aco = ACO(T, num_ants, a, b, rho, Q)
        cost, path = aco.optimize(self)
        return cost, self.cities_nums[path]


class ACO(object):
    def __init__(self, T: int, num_ants: int, a: float, b: float, rho: float, Q: float):
        """
        ACO Problem object

        (Refer to TSP_ACO.optimize() for the description of the attributes)
        One change:
            prob_weight(i, j) -> [0, inf)
        """
        # Parameter initializations
        self.T = T
        self.num_ants = num_ants
        # Hyperparameters (tuned by PSO)
        self.a = a
        self.b = b
        self.rho = rho
        self.Q = Q

        # Results
        self.optimal_cost = math.inf
        self.optimal_path = []

    def update_pheromone(self, tsp_aco: TSP_ACO, ants: list) -> None:
        """
        Update pheromone level for each edge
        :return: None
        """
        for i in range(tsp_aco.N):
            for j in range(i):
                tsp_aco.pheromones[(i, j)] *= 1 - self.rho
                for ant in ants:
                    tsp_aco.pheromones[(i, j)] += ant.pheromone_deposit[(i, j)]

    def optimize(self, tsp_aco: TSP_ACO):
        """
        Perform the optimization, including hyperparameter tuning
        :return: None
        """
        for t in range(self.T):
            # Randomly initialize the ants
            ants = [Ant(tsp_aco, self) for _ in range(self.num_ants)]
            for ant in ants:
                # Each ant constructs a (pseudo-random) Hamiltonian cycle
                ant.explore()
                if ant.total_cost < self.optimal_cost:
                    self.optimal_cost = ant.total_cost
                    self.optimal_path = ant.total_path
                # Amount of pheromone deposited for each ant is updated
                ant.update_pheromone_deposit()
            # Update the pheromones of the TSP_ACO object
            self.update_pheromone(tsp_aco, ants)

        # Return the optimal cost, and the corresponding optimal path
        return self.optimal_cost, self.optimal_path


class Ant(object):
    def __init__(self, tsp_aco: TSP_ACO, aco: ACO):
        """
        Ant object, as a helper for ACO object

        Attributes:
            tsp_aco -- TSP_ACO object that this ant is currently working on
            aco -- ACO object that this ant is currently attached to
            pheromone_deposit -- dict containing the pheromone deposit (by this ant) at each edge ij
                i.e. pheromone_deposit[{i,j}] -> [0, inf)
            current_vertex -- current vertex
            not_allowed -- list of vertices that this ant is not allowed to go to at time t+1
            total_cost -- total distance that the ant has traveled
            total_path -- ordered list of vertices that the ant has traveled
        """
        self.tsp_aco = tsp_aco
        self.aco = aco

        self.pheromone_deposit = defaultdict(lambda: 0.0)

        self.total_vertices = list(range(self.tsp_aco.N))

        # Initialize the starting vertex, uniformly at random!
        initial_vertex = random.randint(0, tsp_aco.N - 1)
        self.current_vertex = initial_vertex
        self.not_allowed = {initial_vertex}
        self.total_cost = 0
        self.total_path = [initial_vertex]

    def prob_weight(self, j, i, a, b):
        """
        helper for self.transition_prob()
        function to be used as a (proportionate) weight when assigning the probability of ant k choosing the edge ij
            i.e. prob_weight(j, i, a, b) -> [0, inf)

        :param j: vertex #
        :param i: vertex #
        :param a: hyperparameter
        :param b: hyperparameter

        :return: unnormalized probability that this ant moves from j to i
        """
        d = self.tsp_aco.distances[(j, i)]
        if d != 0.0:
            return ((self.tsp_aco.pheromones[(j, i)]) ** a) * ((1 / d) ** b)
        else:
            return 0.0

    def transition_prob(self):
        """
        Calculate the transition probabilities of this ant
        :return: list 'prob_list' of probabilities such that
            l[i] = prob that this ant moves from self.current_vertex to i
        """
        prob_list = [0.0 for _ in range(self.tsp_aco.N)]
        a = self.aco.a
        b = self.aco.b

        # Calculate the denominator
        # print("\nNEW START")
        denom = 0.0
        for i in range(self.tsp_aco.N):
            if i not in self.not_allowed:
                denom += self.prob_weight(self.current_vertex, i, a, b)

        for i in range(self.tsp_aco.N):
            if i not in self.not_allowed:
                prob_list[i] = self.prob_weight(self.current_vertex, i, a, b) / denom

        return prob_list

    def move(self) -> None:
        """
        Make this ant move around to output a Hamiltonian cycle
        :return: None
        """
        if len(self.not_allowed) == self.tsp_aco.N:
            raise MoveError("This poor ant can't move to anywhere!")
        # Choose the next vertex, based on the discrete probability distribution
        next_vertex = np.random.choice(self.total_vertices, p=self.transition_prob())
        if next_vertex in self.not_allowed:
            raise MoveError("Why is this ant moving to unauthorized vertex??")
        # Update appropriate quantities
        self.not_allowed.add(next_vertex)
        self.total_cost += self.tsp_aco.distances[(self.current_vertex, next_vertex)]
        self.total_path.append(next_vertex)
        # Move to the next vertex
        self.current_vertex = next_vertex

    def explore(self) -> None:
        """
        Make this ant move around the given TSP to create a Hamiltonian cycle
        :return: None
        """
        for i in range(self.tsp_aco.N - 1):
            self.move()

    def update_pheromone_deposit(self) -> None:
        """
        Update the pheromone deposit, made by this ant, for each edge ij
        :return: None
        """
        for tmp in range(len(self.total_path) - 1):
            i = self.total_path[tmp]
            j = self.total_path[tmp + 1]
            self.pheromone_deposit[(i, j)] = self.aco.Q / self.total_cost
            self.pheromone_deposit[(j, i)] = self.aco.Q / self.total_cost
