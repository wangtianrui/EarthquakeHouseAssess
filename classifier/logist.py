import utils.scores as score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import random


class logistic(object):

    def __init__(self):
        self.model = None

    def train(self, X_train, batch_size, maxStep, log=False):
        if self.model == None:
            self.model = SGDRegressor()

        dim = X_train.shape[1]
        for step in range(maxStep):
            random_index = np.random.choice(X_train, batch_size)
            X = X_train[random_index][:, :dim - 2]
            y = X_train[random_index][:, dim - 1]
            self.model.fit(X, y)
            if step % 1000 == 0 or step == maxStep - 1:
                # mapScore = score.mapScore()
                print(step, "-- mapScore: %f  ,  accuracy: %f", (0, 0))
