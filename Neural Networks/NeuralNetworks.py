import numpy as np
import pandas as pd


def backpropagation(w, z, h, m, n, y, y_hat):
    if h == 3:
        return (y - y_hat) * z[h - 1][m]
    if h == 2:
        prediciton = y - y_hat
        weight = w[h + 1][n][m + 1]
        firstsig = sigmoid(w[h][0][n] + w[h][1][n] * z[h - 1][1] + w[h][2][n] * z[h - 1][2])
        secondsig = 1 - firstsig
        constant = z[h - 1][m]
        return prediciton * weight * firstsig * secondsig * constant
    if h == 1:
        prediction = y - y_hat
        firstweight1 = w[3][1][1]
        firstweight2 = w[h + 1][n][1]
        secondweight1 = w[3][2][1]
        secondweight2 = w[h + 1][n][2]
        firstsig = sigmoid(w[h][0][n] + w[h][1][n] * z[h - 1][1] + w[h][2][n] * z[h - 1][2])
        secondsig = 1 - firstsig
        constant = z[h - 1][m]
        sum = prediction * (firstweight1 * firstweight2 + secondweight1 * secondweight2)
        return sum * firstsig * secondsig * constant

    #prediction = y-y_hat
    #constant = z[h-1][m]
    #sum = prediction
    #for i in range(h-3):
    #    for j in range(len(w[i])/h):
    #sum *= constant
    #return sum

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def stochastic_gradient(train_data):
    nodes = [5, 10, 25, 50, 100]
    w = np.asmatrix(np.zeros(np.zeros(np.zeros(len(nodes[0])))))
    train_dataX = np.asmatrix(train_data.iloc[:, :4].values)
    train_dataY = np.asmatrix(train_data.iloc[:, 4:].values)
    t = 0
    d = 0.01
    for T in range(100):
        random = np.random.permutation(train_data.shape[0] - 1)
        l_0 = 0.0001
        for i in range(train_data.shape[0]-1):
            random_i = random[i]-1
            w = w - l_0 * backpropagation(w,train_dataX[random_i],nodes[0],0,0,train_dataY[random_i], )

            l = l_0 / (1 + (l_0 * t / d))
            l_0 = l
            t += 1

if __name__ == '__main__':
    train_data = pd.read_csv("bank-note/train.csv")
    test_data = pd.read_csv("bank-note/test.csv")
    train_data.columns = ['Variance', 'skewness', 'curtosis', 'entropy', 'y']
    test_data.columns = ['Variance', 'skewness', 'curtosis', 'entropy', 'y']
    train_data["y"] = train_data["y"].replace([0], -1)
    test_data["y"] = test_data["y"].replace([0], -1)
