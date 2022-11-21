import sys
import numpy as np
import pandas as pd


def standard_perceptron(train_data, test_data):
    w = np.asmatrix(np.zeros(4))
    train_dataX = np.asmatrix(train_data.iloc[:, :4].values)
    train_dataY = np.asmatrix(train_data.iloc[:, 4:].values)
    test_dataX = np.asmatrix(test_data.iloc[:, :4].values)
    test_dataY = np.asmatrix(test_data.iloc[:, 4:].values)
    for T in range(10):
        random = np.random.permutation(train_data.shape[0] - 1)
        # finding weight using train data
        for i in range(train_data.shape[0] - 1):
            random_i = random[i] - 1
            wT = np.transpose(w)
            y_hat = train_dataX[random_i] * wT
            ywTx = train_dataY[random_i] * y_hat
            if ywTx <= 0:
                w = w + train_dataY[random_i] * train_dataX[random_i]
        # computing prediction using weight we found with train data
        error = 0
        for j in range(test_data.shape[0] - 1):
            wT = np.transpose(w)
            y_hat = test_dataX[j] * wT
            ywTx = test_dataY[j] * y_hat
            if ywTx <= 0:
                error += 1
        size = test_data.shape[0] - 1
        average_error = error / size
        print("T = " + str(T + 1) + ", W = " + str(w) + ", average prediction error = " + str(average_error))

    return


def voted_perceptron(train_data, test_data):
    w = np.asmatrix(np.zeros(4))
    C = [0]
    m = 0
    train_dataX = np.asmatrix(train_data.iloc[:, :4].values)
    train_dataY = np.asmatrix(train_data.iloc[:, 4:].values)
    test_dataX = np.asmatrix(test_data.iloc[:, :4].values)
    test_dataY = np.asmatrix(test_data.iloc[:, 4:].values)
    for T in range(10):
        for i in range(train_data.shape[0] - 1):
            wT = np.transpose(w[m])
            y_hat = train_dataX[i] * wT
            ywTx = train_dataY[i] * y_hat
            _w = w[m] + train_dataY[i] * train_dataX[i]
            if ywTx <= 0 and (w[m] != _w).any():
                w = np.vstack([w, _w])
                m = m + 1
                C.append(1)
            else:
                C[m] = C[m] + 1
        for j in range(len(C)):
            print("Weight Vector: " + str(w[j]) + ", counts: " + str(C[j]))
        error = 0
        for k in range(test_data.shape[0] - 1):
            temp = 0
            for i in range(len(C)):
                wT = np.transpose(w[i])
                y_hat = np.sign(test_dataX[k] * wT)
                temp += C[i] * y_hat
            prediction = np.sign(temp)
            if np.sign(test_dataY[k]) != prediction:
                error += 1
        size = test_data.shape[0] - 1
        average_error = error / size
        print("average prediction error = " + str(average_error))


def averaged_perceptron(train_data, test_data):
    w = np.asmatrix(np.zeros(4))
    a = np.asmatrix(np.zeros(4))
    train_dataX = np.asmatrix(train_data.iloc[:, :4].values)
    train_dataY = np.asmatrix(train_data.iloc[:, 4:].values)
    test_dataX = np.asmatrix(test_data.iloc[:, :4].values)
    test_dataY = np.asmatrix(test_data.iloc[:, 4:].values)
    for T in range(10):
        for i in range(train_data.shape[0] - 1):
            wT = np.transpose(w)
            y_hat = train_dataX[i] * wT
            ywTx = train_dataY[i] * y_hat
            if ywTx <= 0:
                w = w + train_dataY[i] * train_dataX[i]
            a = a + w
        error = 0
        for j in range(test_data.shape[0] - 1):
            aT = np.transpose(a)
            prediction = test_dataX[j] * aT
            if np.sign(prediction) != np.sign(test_dataY[j]):
                error += 1
        size = test_data.shape[0] - 1
        average_error = error / size
        print("T = " + str(T + 1) + ", W = " + str(w) + ", average prediction error = " + str(average_error))


if __name__ == '__main__':
    train_data = pd.read_csv("bank-note/train.csv")
    test_data = pd.read_csv("bank-note/test.csv")
    train_data.columns = ['Variance', 'skewness', 'curtosis', 'entropy', 'y']
    test_data.columns = ['Variance', 'skewness', 'curtosis', 'entropy', 'y']
    if sys.argv[1] == "standard":
        print("RUNNING STANDARD PERCEPTRON")
        standard_perceptron(train_data, test_data)
    elif sys.argv[1] == "voted":
        print("RUNNING VOTED PERCEPTRON")
        voted_perceptron(train_data, test_data)
    elif sys.argv[1] == "averaged":
        print("RUNNING AVERAGED PERCEPTRON")
        averaged_perceptron(train_data, test_data)
