import numpy as np
from Kalman import Kalman

if __name__ == '__main__':

    kf = Kalman(7, 2)

    fcst = [9., 3., 6., 7., 11., 0.4, 4.2]
    obs = [9.5, 2.4, 6.1, 7.7, 12.1, 1., 3.8]
    model = obs + np.random.normal(0.0, 1.1, size = len(obs))

    for ij in range(2):

        # Inform
        kf.dump_members(ij=ij)

        # Get the observations matrix based on old model prediction
        kf.calculate_obs_matrix(model[ij])

        # Calculate variance and covariance of the dataset
        kf.calculate_variance()
        kf.calculate_covariance()

        # Calculate Kalman gain
        kf.calculate_kalman_gain()

        # Update the state vector based on the difference between true value and model previous forecast 
        kf.update_state_vector(obs[ij], model[ij])

        # Get a new, corrected value of current forecast, considering the previous forecast value
        newVal = kf.correct_measurement(model[ij], fcst[ij], onlyPos=False)

        # Update working matrices with new input
        kf.update_matrices(obs[ij], model[ij])

        # Inform
        kf.dump_members(ij=ij)

        # Show the corrected value
        print('Corrected value: {}'.format(newVal))



    