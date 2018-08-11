from classifier.LGB import LGB
from utils.read_data import DataLoader
import numpy as np
import random
import utils.pre_functions as pf
from classifier.LGB import LGB

if __name__ == "__main__":
    dataLoader = DataLoader()
    train_data, predict_data = dataLoader.loadData()
    print(train_data.shape, predict_data.shape)
    train_data, predict_data = pf.dealWithNan(train_data, predict_data)
    print(train_data.shape, predict_data.shape)
    train_data, predict_data = pf.pca_function(train_data, predict_data, 13)
    print(train_data.shape, predict_data.shape)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    print(X_train.shape)
    print(y_train.shape)
    params = {

    }
    lgb_origin = LGB(params=params)
    lgb_origin.train(X_train=X_train, y_train=y_train, X_test=X_train, y_test=y_train, log=True)
    y_predict = lgb_origin.predict(predict_data)
    # dataLoader.save_to_commit(y_predict)
