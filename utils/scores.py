import numpy as np
from sklearn.metrics import accuracy_score


def mapScore(y_predict, y_label):
    y_predict = np.array(y_predict)  # batch x 2
    y_label = np.array(y_label).reshape(-1, 1)

    batch_size, dim = y_predict.shape  # dim = 2

    equal_map = (y_predict == y_label).astype(int)  # batch x dim
    denominator = np.tile(np.array([1.0, 2.0]), (batch_size, 1))

    y_divide_deno = np.ones(shape=(batch_size, dim)) / denominator * equal_map
    score = np.mean(np.sum(y_divide_deno, axis=1))
    return score


def accuracy(y_predict, y_label):
    return accuracy_score(y_predict, y_label)
