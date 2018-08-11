import pandas as pd
import numpy as np
import os
from utils.read_data import DataLoader

father_path = os.path.abspath(os.path.join(os.path.dirname(""), "data"))
print(father_path)
predict1 = pd.read_csv(os.path.join(father_path, "pca10（0.79196）.csv"))
predict2 = pd.read_csv(os.path.join(father_path, "sample_submit（0.78854）.csv"))
predict3 = pd.read_csv(os.path.join(father_path, "save（pca12,150leaves）.csv"))
predict4 = pd.read_csv(os.path.join(father_path, "100leaves.csv"))

predict1 = np.array(predict1)
predict2 = np.array(predict2)
predict3 = np.array(predict3)
predict4 = np.array(predict4)

num, dim = predict1.shape

new_predict = np.zeros((num, dim))

for index in range(num):
    dict_y1 = [0, 0, 0, 0]
    dict_y2 = [0, 0, 0, 0]

    dict_y1[int(predict1[index, 0])] += 1
    dict_y2[int(predict1[index, 1])] += 1

    dict_y1[int(predict2[index, 0])] += 1
    dict_y2[int(predict2[index, 1])] += 1

    dict_y1[int(predict3[index, 0])] += 1
    dict_y2[int(predict3[index, 1])] += 1

    dict_y1[int(predict4[index, 0])] += 1
    dict_y2[int(predict4[index, 1])] += 1

    # print(predict1[index])
    # print(predict2[index])
    # print(predict3[index])
    # print(predict4[index])
    # print(dict_y1)
    # print(dict_y2)

    new_predict[index, 0] += np.argmax(dict_y1)
    new_predict[index, 1] += np.argmax(dict_y2)

print(new_predict[-1])

datawriter = DataLoader()
datawriter.save_to_commit(new_predict)
