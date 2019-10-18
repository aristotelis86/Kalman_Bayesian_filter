"""
Defining the basic Bayesian Inference Corrector.
Can be used as standalone tool as well as in
combination with the Kalman filter.
"""

import numpy as np

class Bayes:
    def __init__(self, history):
        if history < 1:
            raise ValueError('History length is too small')

        # Set the bare minimum info needed
        self.history = history
        self.obsValues = np.zeros(history)
        self.modValues = np.zeros(history)

        self.correctionType = 'none'
        self.avgObs = None
        self.sigmaObs = None
        self.sigmaError = None
        self.sigmaCorrection = None
        


