import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.special import expit


def tanh(x):
    np.seterr(divide='ignore', invalid='ignore')
    return (expit(x) - expit(-x)) / (expit(x) + expit(-x))


def tanh_derivitive(x):
    np.seterr(divide='ignore', invalid='ignore')

    x = x[0, :]
    x = np.diag(x)
    a = (1 - (sigmoid(x)**2))
    return a


def sigmoid(x):
    
    return 1/(1 + expit(-x))


def sigmoid_derivitive(x):
    x = x[0, :]
    x = np.diag(x)
    return (sigmoid(x) * (1-sigmoid(x)))


def normalize_data_in_range_minus_1_to_1(data):
    return (2 * (data - data.min())/(data.max() - data.min())) - 1


def get_data():
    df = pd.read_csv("Data.csv")
    data = df.to_numpy()
    data.astype(np.float)
    data = normalize_data_in_range_minus_1_to_1(data)
    data1 = data[0:]
    data2 = np.append(data[1:], np.array([0] * 1))
    data3 = np.append(data[2:], np.array([0]*2))
    data4 = np.append(data[3:], np.array([0]*3))
    data5 = np.append(data[4:], np.array([0]*4))

    y = np.append(data[11:], np.array([0]*11))

    x = np.column_stack((data1, data2, data3, data4, data5))
    x = x[0:-11, :]
    y = y[0:-11]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    return x_train, y_train, x_test, y_test
