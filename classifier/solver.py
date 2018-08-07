import utils.scores as score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import numpy as np
import random
import xgboost as xgb


class Solver(object):

    def __init__(self):
        self.model = None
        self.ss = StandardScaler()
        self.flag = None

    def logisticTrain(self, X_train, batch_size, maxStep, log=False):
        if self.model == None:
            self.model = SGDClassifier()
            self.ss = StandardScaler()
            self.flag = "logistic"
        dim = X_train.shape[1]
        X_all = X_train[:, :dim - 1]
        y_all = X_train[:, dim - 1]
        X_all = self.ss.fit_transform(X_all)
        for step in range(maxStep):
            random_index = np.random.choice(X_train.shape[0], batch_size)
            X = X_all[random_index]
            y = y_all[random_index]
            self.model.fit(X, y)
            if log:
                if step % 50000 == 0 or step == maxStep - 1:
                    random_index = np.random.choice(X_train.shape[0], batch_size)
                    X = X_all[random_index]
                    y = y_all[random_index]
                    y_pred = self.model.predict(X)
                    # mapScore = score.mapScore()
                    # print(step, "-- mapScore: %f  ,  accuracy: %f", (0, 0))
                    print(step, "-- accuracy: %f", (score.accuracy_score(y, y_pred)))

    def randomForest(self, X_train, log=False):
        if self.model == None:
            self.model = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=10)
            self.ss = StandardScaler()
            self.flag = "randomForest"
        num, dim = X_train.shape
        X_all = X_train[:, :dim - 1]
        y_all = X_train[:, dim - 1]
        X_all = self.ss.fit_transform(X_all)
        self.model.fit(X_all, y_all)
        if log:
            random_index = np.random.choice(X_train.shape[0], 500)
            X = X_all[random_index]
            y = y_all[random_index]
            y_pred = self.model.predict(X)
            # mapScore = score.mapScore()
            # print(step, "-- mapScore: %f  ,  accuracy: %f", (0, 0))
            print("-- accuracy: %f", (score.accuracy_score(y, y_pred)))

    def lgb(self, X_train, params, num_boost_round, log=False):

        self.flag = "lgb"
        num, dim = X_train.shape
        X_all = X_train[:, :dim - 1]
        y_all = X_train[:, dim - 1]
        X_all = self.ss.fit_transform(X_all)

        random_index = np.random.choice(X_train.shape[0], 500)
        X = X_all[random_index]
        y = y_all[random_index]

        lgb_train = lgb.Dataset(X_all, y_all)
        lgb_eval = lgb.Dataset(X, y)
        self.model = lgb.train(params,
                               lgb_train,
                               num_boost_round=num_boost_round,
                               valid_sets=lgb_eval,
                               early_stopping_rounds=500,
                               verbose_eval=False)
        if log:
            random_index = np.random.choice(X_train.shape[0], 500)
            X = X_all[random_index]
            y = y_all[random_index]
            softmax = self.model.predict(X, num_iteration=self.model.best_iteration)
            first = np.argmax(softmax, axis=1)
            print("accuracy:%f", (score.accuracy_score(first, y)))
            softmax[range(X.shape[0]), first] = 0
            second = np.argmax(softmax, axis=1).reshape(-1, 1)
            first = first.reshape(-1, 1)
            y_pred = np.append(first, second, axis=1)
            print("map score: %f", (score.mapScore(y_pred, y)))
            print("最佳迭代次数：", self.model.best_iteration)

    def xgboost(self, X_train, params, num_boost_round, log=False):
        self.flag = "xgb"
        num, dim = X_train.shape
        X_all = X_train[:, :dim - 1]
        y_all = X_train[:, dim - 1]
        X_all = self.ss.fit_transform(X_all)

        random_index = np.random.choice(X_train.shape[0], 500)
        X = X_all[random_index]
        y = y_all[random_index]

        xgb_train = xgb.DMatrix(X_all, label=y_all)
        xgb_test = xgb.DMatrix(X, label=y)

        watchlist = [(xgb_train, "train"), (xgb_test, "test")]

        self.model = xgb.train(
            params,
            xgb_train,
            num_boost_round,
            watchlist
        )
        if log:
            random_index = np.random.choice(X_train.shape[0], 500)
            X = X_all[random_index]
            y = y_all[random_index]
            softmax = self.model.predict(X, num_iteration=self.model.best_iteration)
            first = np.argmax(softmax, axis=1)
            print("accuracy:%f", (score.accuracy_score(first, y)))
            softmax[range(X.shape[0]), first] = 0
            second = np.argmax(softmax, axis=1).reshape(-1, 1)
            first = first.reshape(-1, 1)
            y_pred = np.append(first, second, axis=1)
            print("map score: %f", (score.mapScore(y_pred, y)))
            print("最佳迭代次数：", self.model.best_iteration)

    def predict(self, X):
        if self.flag == "lgb" or self.flag == "xgb":
            X = self.ss.fit_transform(X)
            softmax = self.model.predict(X, num_iteration=self.model.best_iteration)
            first = np.argmax(softmax, axis=1)
            softmax[range(X.shape[0]), first] = 0
            second = np.argmax(softmax, axis=1).reshape(-1, 1)
            first = first.reshape(-1, 1)
            y_pred = np.append(first, second, axis=1)
            return y_pred
