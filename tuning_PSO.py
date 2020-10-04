import math
import numpy as np


class PSO(object):
    def __init__(self, hyperparam, T):
        """
        PSO object, to be used for (4-D) hyperparameter tuning of ACO object

        Attributes:
            hyperparam -- 4-dimensional hyperparameter vector to be tuned
                i.e. (a, b, Q, rho) -- hyperparameters to be tuned
            T -- max running time of this PSO
        """
        # Initialize
        self.hyperparam = hyperparam
        self.T = T


class Particle(object):
    def __init__(self, hyperparam0, fitness):
        """
        Particle object, as a helper for PSO object

        Attributes:
            x -- position of this particle
            v -- velocity of this particle
            y -- loss function value at x
            y_best -- lowest loss value achieved
            x_best -- corresponding position
            fitness -- fitness function
                i.e. fitness(hyperparam) -> [0, inf)
        """
        # Initialize positions and velocities
        self.x = hyperparam0
        self.y = math.inf
        self.v = np.random.uniform(-1, 1, size=(4,)).tolist()

        # Saving the best ones
        self.y_best = math.inf
        self.x_best = []

        # Fitness function - total cost from ACO
        self.fitness = fitness

        # Hyperparameters for the PSO
        self.c1 = 2
        self.c2 = 2

    def check_best(self):
