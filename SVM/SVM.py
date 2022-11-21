import sys
import numpy as np
import pandas as pd


def a_svm_stochastic_sub_gradient(train_data, C):
    w = np.asmatrix(np.zeros(4))
    train_dataX = np.asmatrix(train_data.iloc[:, :4].values)
    train_dataY = np.asmatrix(train_data.iloc[:, 4:].values)
    N = train_data.shape[0]
    t = 0
    a = 0.5
    for T in range(100):
        random = np.random.permutation(train_data.shape[0] - 1)
        l_0 = 0.1
        for i in range(train_data.shape[0] - 1):
            random_i = random[i] - 1
            margin = 1 - train_dataY[random_i] * train_dataX[random_i] * w.T
            if max(0, margin) == 0:
                w = (1 - l_0) * w
            else:
                w = w - l_0 * (w + l_0 * C * N * train_dataY[random_i] * train_dataX[random_i])
            l = l_0 / (1 + (l_0 * t / a))
            l_0 = l
            t += 1
    return w


def b_svm_stochastic_sub_gradient(train_data, C):
    w = np.asmatrix(np.zeros(4))
    train_dataX = np.asmatrix(train_data.iloc[:, :4].values)
    train_dataY = np.asmatrix(train_data.iloc[:, 4:].values)
    N = train_data.shape[0]
    t = 0
    for T in range(100):
        random = np.random.permutation(train_data.shape[0] - 1)
        l_0 = 0.1
        for i in range(train_data.shape[0] - 1):
            random_i = random[i] - 1
            margin = 1 - train_dataY[random_i] * train_dataX[random_i] * w.T
            if max(0, margin) == 0:
                w = (1 - l_0) * w
            else:
                w = w - l_0 * (w + l_0 * C * N * train_dataY[random_i] * train_dataX[random_i])
            l = l_0 / (1 + t)
            l_0 = l
            t += 1
    return w

def error(w, data):
    dataX = np.asmatrix(data.iloc[:, :4].values)
    dataY = np.asmatrix(data.iloc[:, 4:].values)
    error = 0
    for i in range(data.shape[0] - 1):
        prediction = dataX[i] * w.T
        if np.sign(prediction) != np.sign((dataY[i])):
            error += 1

    return error / dataX.shape[0]

def dual_svm_stochastic_sub_gradient(train_data, C):
    w = np.asmatrix(np.zeros(4))
    train_dataX = np.asmatrix(train_data.iloc[:, :4].values)
    train_dataY = np.asmatrix(train_data.iloc[:, 4:].values)
    b = 1
    random = np.random.permutation(train_data.shape[0] - 1)
    for i in range(train_data.shape[0] - 1):
        random_i = random[i] - 1
        for j in range(train_data.shape[0] - 1):
            random_j = random[j] - 1
            prediciton = train_dataX[random_i] * w.T + b
            if np.sign(prediciton) != np.sign(train_dataY[random_i]):
                w = C * train_dataY[random_i] * train_dataX[random_i]
                b = train_dataY[random_j] - train_dataX[random_j] * w.T

    return w




if __name__ == '__main__':
    train_data = pd.read_csv("bank-note/train.csv")
    test_data = pd.read_csv("bank-note/test.csv")
    train_data.columns = ['Variance', 'skewness', 'curtosis', 'entropy', 'y']
    test_data.columns = ['Variance', 'skewness', 'curtosis', 'entropy', 'y']
    train_data["y"] = train_data["y"].replace([0], -1)
    test_data["y"] = test_data["y"].replace([0], -1)
    C = [100 / 873, 500 / 873, 700 / 873]
    print("Primal")
    print("with improved update learning rate")
    w = a_svm_stochastic_sub_gradient(train_data, C[0])
    print("Error for training with C = 100/873 : " + str(error(w, train_data)))
    print("Error for testing with C = 100/873 : " + str(error(w, test_data)))
    w = a_svm_stochastic_sub_gradient(train_data, C[1])
    print("Error for training with C = 500/873 : " + str(error(w, train_data)))
    print("Error for testing with C = 500/873 : " + str(error(w, test_data)))
    w = a_svm_stochastic_sub_gradient(train_data, C[2])
    print("Error for training with C = 700/873 : " + str(error(w, train_data)))
    print("Error for testing with C = 700/873 : " + str(error(w, test_data)))
    print("with update learning rate")
    w = b_svm_stochastic_sub_gradient(train_data, C[0])
    print("Error for training with C = 100/873 : " + str(error(w, train_data)))
    print("Error for testing with C = 100/873 : " + str(error(w, test_data)))
    w = b_svm_stochastic_sub_gradient(train_data, C[1])
    print("Error for training with C = 500/873 : " + str(error(w, train_data)))
    print("Error for testing with C = 500/873 : " + str(error(w, test_data)))
    w = b_svm_stochastic_sub_gradient(train_data, C[2])
    print("Error for training with C = 700/873 : " + str(error(w, train_data)))
    print("Error for testing with C = 700/873 : " + str(error(w, test_data)))
    print("Dual")
    w = dual_svm_stochastic_sub_gradient(train_data, C[0])
    print("Error for training with C = 100/873 : " + str(error(w, train_data)))
    print("Error for testing with C = 100/873 : " + str(error(w, test_data)))
    w = dual_svm_stochastic_sub_gradient(train_data, C[1])
    print("Error for training with C = 500/873 : " + str(error(w, train_data)))
    print("Error for testing with C = 500/873 : " + str(error(w, test_data)))
    w = dual_svm_stochastic_sub_gradient(train_data, C[2])
    print("Error for training with C = 700/873 : " + str(error(w, train_data)))
    print("Error for testing with C = 700/873 : " + str(error(w, test_data)))


