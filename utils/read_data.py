import pandas as pd
import numpy as np
import os


class DataLoader(object):

    def __init__(self):
        self.train_data = None
        self.predict_data = None

    def loadData(self):

        father_tree_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        train_data_file = os.path.join(father_tree_path, "data/train.csv")
        predict_data_file = os.path.join(father_tree_path, "data/test.csv")
        # predict_data_file = "../data/test.csv"
        if os.path.exists(train_data_file):
            self.train_data = pd.read_csv(train_data_file)
        else:
            print(train_data_file, "is not exist")
        if os.path.exists(predict_data_file):
            self.predict_data = pd.read_csv(predict_data_file)
        else:
            print(predict_data_file, "is not exist")
        return self.train_data, self.predict_data

    def getNumpyDatas(self, flag="all"):

        if self.train_data == None or self.predict_data == None:
            self.loadData()
        if flag == "train":
            return np.array(self.train_data)
        elif flag == "predict":
            return np.array(self.predict_data)
        else:
            return np.array(self.train_data), np.array(self.predict_data)
