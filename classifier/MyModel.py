from sklearn.model_selection import GridSearchCV
from abc import abstractmethod
import numpy as np


class MyModel:
    def __init__(self):
        """
        初始化模型
        :param params: 模型参数dict
        """
        self.model = None
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_test=None, y_test=None, step=100, log=False):
        """
        训练模型
        :param X_train: 训练集features
        :param y_train: 训练集labels
        :param step: 步数
        :param log: 是否输出log
        :return:
        """
        if self.model == None:
            print("请在__init__中实现model")
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        预测
        :param X_test:
        :return:
        """
        pass

    @abstractmethod
    def gridSearchCV(self, grid_params, X, y, cv=5):
        """
        参数测试
        :param grid_params:
        :param X:
        :param y:
        :param cv:
        :return:
        """
        searcher = GridSearchCV(estimator=self.model, param_grid=grid_params, cv=cv)
        searcher.fit(X, y)
        print(searcher.cv_results_)
        pass

    def scores(self, y_pred, y_label, kind="classify"):
        if kind == "classify":
            print("this model is a classify model")
            TP = float(np.sum(y_pred * y_label))
            TN = np.sum((1 - y_pred) * (1 - y_label))
            FN = np.sum((1 - y_pred) * y_label)
            FP = np.sum(y_pred * (1 - y_label))
            accuracy = (TP + TN) / (TP + TN + FN + FP)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            F_measure = 2 / ((1 / precision) + (1 / recall))
            print("accuracy:%f ; recall:%f ; precision:%f ; F-measure:%f" % (accuracy, recall, precision, F_measure))
