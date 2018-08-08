import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def dealWithNan(X_train, X_predit):
    X_train.dropna(axis=0, how="all")

    # char 2 num
    X_train, X_predit = changeChar2Num(X_train, X_predit, "land_condition")
    X_train, X_predit = changeChar2Num(X_train, X_predit, "foundation_type")
    X_train, X_predit = changeChar2Num(X_train, X_predit, "roof_type")
    X_train, X_predit = changeChar2Num(X_train, X_predit, "ground_floor_type")
    X_train, X_predit = changeChar2Num(X_train, X_predit, "position")

    # 均值填充
    X_train = fillNanWithMean(X_train)
    X_predit = fillNanWithMean(X_predit)
    return np.array(X_train), np.array(X_predit)


def changeNan(dict, x):
    # print(dict[x])
    return dict[x]


def changeChar2Num(X_train, X_predit, row_name):
    # 得到每列所有不同元素的值  data["type"].unique(),返回numpylist
    unique_list = np.append(X_train[row_name].unique(), X_predit[row_name].unique())
    dict = {}
    number = 0
    for index in unique_list:
        dict[index] = number
        number += 1
    X_train[row_name] = X_train[row_name].apply(lambda x: changeNan(dict, x))
    X_predit[row_name] = X_predit[row_name].apply(lambda x: changeNan(dict, x))
    return X_train, X_predit


def fillNanWithMean(df):
    for column in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[column].mean()
        df[column].fillna(mean_val, inplace=True)
    return df


def pca_function(X_train, X_test, n_components):
    train_num, train_dim = X_train.shape
    # test_num, test_dim = X_test.shape
    # pca = PCA(n_components)
    X_train_all = X_train[:, :train_dim - 1]
    y_train_all = X_train[:, train_dim - 1]
    #
    # X_train_pca = pca.fit_transform(X_train_all)
    # X_train = np.append(X_train_pca, y_train_all)
    #
    # X_test_all = X_test[:, :test_dim - 2]
    # y_test_all = X_test[:, test_dim - 1]
    # X_test_pca = pca.fit_transform(X_test_all)
    # X_test = np.append(X_test_pca, y_test_all)
    data_all = np.append(X_train_all, X_test, axis=0)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(data_all)
    print(X_pca[:train_num].shape,y_train_all.shape)
    X_train = np.append(X_pca[:train_num], y_train_all.reshape(-1,1), axis=1)
    X_test = X_pca[train_num:]
    # print(data_all.shape)

    return X_train, X_test
