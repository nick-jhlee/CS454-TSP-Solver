class PSO(object):
    def __init__(self, a0, b0, Q0, rho0):
        """
        PSO object, to be used for (4-D) hyperparameter tuning of ACO object

        Attributes:
            (a, b, Q, rho) -- hyperparameters to be tuned
        """
        # Initialize
        self.a = a0
        self.b = b0
        self.Q = Q0
        self.rho = rho0