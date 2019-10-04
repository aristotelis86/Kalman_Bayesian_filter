import numpy as np
from Kalman import Kalman

if __name__ == '__main__':

    kf = Kalman(7, 2)

    fcst = [9., 3., 6., 7., 11., 0.4, 4.2]
    obs = [9.5, 2.4, 6.1, 7.7, 12.1, 1., 3.8]
    model = obs + np.random.normal(0.0, 1.1, size = len(obs))

    kf.train_me(obs, model)
    
    # for ij in range(1):

    #     # Inform
    #     kf.dump_members(ij='Starting iterantion {:d}'.format(ij))

    #     kf.train_me(obs, model)


        # # Get the observations matrix based on old model prediction
        # kf.calculate_obs_matrix(model[ij])

        # # Calculate variance and covariance of the dataset
        # kf.calculate_variance()
        # kf.calculate_covariance()

        # # Calculate Kalman gain
        # kf.calculate_kalman_gain()

        # # Update the state vector based on the difference between true value and model previous forecast 
        # kf.update_state_vector(obs[ij], model[ij])

        # # Get a new, corrected value of current forecast, considering the previous forecast value
        # newVal = kf.correct_measurement(model[ij], fcst[ij], onlyPos=False)

        # # Update working matrices with new input
        # kf.update_matrices(obs[ij], model[ij])

        # # Inform
        # kf.dump_members(ij=ij)

        # # Show the corrected value
        # print('Corrected value: {}'.format(newVal))

    # x_history = np.zeros((2,7))

    # x_history[0,0] = 1.0
    # x_history[1,0] = 2.5

    # x_history[0,1] = 1.25
    # x_history[1,1] = 2.68

    # x_history[0,2] = 1.32
    # x_history[1,2] = 2.89

    # x_history[0,3] = 2.0
    # x_history[1,3] = 2.8

    # x_history[0,4] = 1.0
    # x_history[1,4] = 2.9

    # x_history[0,5] = 0.54
    # x_history[1,5] = 3.1

    # x_history[0,6] = 0.0
    # x_history[1,6] = 0.2

    # #print(x_history)

    # S1 = np.zeros((2,1))

    # for ij in range(6):
    #     S1[0] += x_history[0,ij+1] - x_history[0,ij]
    #     S1[1] += x_history[1,ij+1] - x_history[1,ij]

    # S1 /= 7
    # #print(S1)

    # S2 = np.zeros((2,1))
    # theSum = np.zeros((2,2))
    # for ij in range(6):
    #     for jj in range(2):
    #         S2[jj] = x_history[jj, ij+1] - x_history[jj, ij] - S1[jj]

    #     theSum += np.dot(S2, S2.T)



    # theSum /= 6
    # print(theSum)
    # print('')
    # print(np.cov(x_history))

    