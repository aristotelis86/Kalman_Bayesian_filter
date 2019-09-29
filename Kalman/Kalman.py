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
                self.F = np.array(F)
            else:
                raise ValueError('Transition matrix F is has the wrong dimensions.')
        
        # Get the bias history matrix
        if not yV:
            self.yV = 10.0 * np.ones(self.history)
        else:
            if yV.shape[0] == self.history:
                self.yV = np.array(yV)
            else:
                raise ValueError('Bias history array yV has the wrong dimensions.')

        # Get the history state matrix
        if not x_mat:
            self.x_mat = 10.0 * np.ones((self.dim, self.history + 1))

        # Get the covariance matrix
        if not P:
            self.P = 10.0 * np.ones((self.dim, self.dim))
        else:
            if P.shape[0] == self.dim:
                self.P = np.array(P)
            else:
                raise ValueError('Covariance matrix P has the wrong dimensions.')
        
        # Get the state vector
        if not x:
            self.x = 10.0 * np.ones(self.dim)
        else:
            if x.shape[0] == self.dim:
                self.x = np.array(x)
            else:
                raise ValueError('State vector x has the wrong dimensions.')
        
        # Hope it's classic Kalman, but you never know
        self.classic = True
        if exp_var:
            print('Switching to Information Geometry Kalman filter...')
            print('Variance for data is provided explicitly as: {}'.format(exp_var))
            self.classic = False
            self.variance = exp_var
        else:
            self.variance = 0.0 # self.calculate_variance()

        # Get the covariance of state matrix
        self.xV = np.zeros((self.dim, self.history))
        for irow in range(self.dim):
            for icol in range(self.history):
                self.xV[irow, icol] = self.x_mat[irow, icol + 1] - self.x_mat[irow, icol]

        self.covariance = 0.0 #self.calculate_covariance()

        # Initialise other relevant matrices
        self.H = np.zeros(self.dim)     # Observations matrix
        self.KG = np.zeros_like(self.x) # Kalman gain 

    def dump_members(self, ij = None):
        """
        Defining the "print" method for 
        debugging and informative purposes.
        """
        print('--------------------------')
        print('     Kalman Instance      ')
        if not (ij is None):
            print('(iteration = {})'.format(ij))
        print('Classic? {}'.format(self.classic))
        print('History: {}'.format(self.history))
        print('Dimension: {}'.format(self.dim))
        print('F: {}'.format(self.F))
        print('yV: {}'.format(self.yV))
        print('x_mat: {}'.format(self.x_mat))
        print('X: {}'.format(self.x))
        print('P: {}'.format(self.P))
        print('xV: {}'.format(self.xV))
        print('Var: {}'.format(self.variance))
        print('Covar: {}'.format(self.covariance))
        print('KG: {}'.format(self.KG))
        print('**************************')



    def calculate_variance(self):
        """
        Method to calculate the variance of the bias.
        """
        return np.var(self.yV)

    def get_variance(self):
        return self.variance

    def calculate_covariance(self):
        return np.cov(self.xV)

    def calculate_obs_matrix(self, model):
        for ij in range(self.dim):
            self.H[ij] = model * (ij - 1)
        return 

    def calculate_kalman_gain(self):
        """
        Method to calculate the Kalman gain.
        """
        P1 = self.P + self.covariance

        Q = np.dot(np.dot(self.H, P1), self.H.T)

        self.KG = np.dot(P1, self.H.T) / (Q + self.variance)

    def update_state_vector(self, obs, model):
        dif = obs - model - np.dot(self.H, self.x)
        self.x += self.KG * dif 
    
    def correct_measurement(self, old_model, measurement, hard_limit = None, onlyPos = True):
        """
        Make an attempt to correct the measurement or model output.
        """
        pre = 0.0
        for ij in range(self.dim):
            pre += self.x[ij] * old_model ** (ij - 1)

        if hard_limit:
            if pre > hard_limit:
                pre = measurement
            else:
                pre += measurement
        else:
            pre += measurement

        if onlyPos:
            if pre < 0.0:
                pre = 0.0
        
        return pre

    def update_matrices(self, obs, model):

        myKG = np.zeros((self.KG.shape[0],1))
        myH = np.zeros((self.H.shape[0],1))

        myKG[:,0] = self.KG
        myH[:,0] = self.H

        D = np.eye(self.dim) - np.matmul(myKG, myH.T)

        self.P = np.dot(D, self.P)

        self.yV[0:self.history - 1] = self.yV[1:self.history]
        self.yV[self.history - 1] = obs - model
        for ij in range(self.dim):
            self.yV[self.history - 1] -= self.x[ij] * model ** (ij - 1)

        self.x_mat[:, 0:self.history - 1] = self.x_mat[:, 1:self.history]
        self.x_mat[:, self.history - 1] = self.x[:]










        





        

