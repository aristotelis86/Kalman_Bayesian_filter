"""
Defining the basic Kalman filtered
adjusted for correcting meteorological
data (ie no explicit system dynamics).
"""

import numpy as np

class Kalman:
    def __init__(self, history, dim, F = None, yV = None, x_mat = None, P = None, x = None, exp_var = None):
        
        if history < 1 or dim < 1:
            raise ValueError('Improper value entered for history length and/or dimension of observation matrix...')
            
        if dim < 2:
            print('Caution! Low accuracy due to the order of the order of the observation matrix.')

        # Set the bare minimum info
        self.history = history
        self.dim = dim

        # Get the transition matrix
        if not F:
            self.F = np.eye(self.dim)
        else:
            if F.shape[0] == self.dim:
                self.F = F
            else:
                raise ValueError('Transition matrix F is has the wrong dimensions.')
        
        # Get the bias history matrix
        if not yV:
            self.yV = 10.0 * np.ones(self.history)
        else:
            if yV.shape[0] == self.history:
                self.yV = yV
            else:
                raise ValueError('Bias history array yV has the wrong dimensions.')

        # Get the history state matrix
        if not x_mat:
            self.x_mat = 10.0 * np.ones((self.dim, self.history + 1))

        # Get the covariance matrix
        if not P:
            self.P = np.zeros((self.dim, self.dim))
        else:
            if P.shape[0] == self.dim:
                self.P = P
            else:
                raise ValueError('Covariance matrix P has the wrong dimensions.')
        
        # Get the state vector
        if not x:
            self.x = 10.0 * np.ones(self.dim)
        else:
            if x.shape[0] == self.dim:
                self.x = x
            else:
                raise ValueError('State vector x has the wrong dimensions.')
        
        # Hope it's classic Kalman, but you never know
        self.classic = True
        if exp_var:
            print('Switching to Information Geometry Kalman filter...')
            print('Variance for data is provided explicitly as: {}'.format(exp_var))
            self.classic = False
            self.variance = exp_var

        

        

