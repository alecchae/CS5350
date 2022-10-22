import numpy as np
import pandas as pd
import random

def batch_gradient_descent(train_data, iteration=100):
    r=1
    weights1 = np.asmatrix(np.zeros(7))
    weights0 = np.asmatrix(np.zeros(7))
    train_dataX = np.asmatrix(train_data.iloc[:, :7].values)
    train_dataY = np.asmatrix(train_data.iloc[:, 7:].values)
    while True:
        for t in range(iteration):
            wT = np.transpose(weights0)
            wTx = np.matmul(train_dataX, wT)
            error = np.subtract(train_dataY, wTx)
            errorT = np.transpose(error)
            gradient = errorT * train_dataX
            cost = np.sum(np.square(error)) * 0.5
            print("at step" + str(t))
            weights1 = weights0 - r * gradient
            _wT = np.transpose(weights1)
            _wTx = train_dataX * _wT
            _error = train_dataY - _wTx
            cost = np.sum(np.square(error)) * 0.5
            if np.linalg.norm(weights1 - weights0) < 0.000001:
                print("cost: " + str(cost))
                print("Final learned weight vector: " + str(weights1))
                print("Final learning rate: " + str(r))
                return
            print("cost: " + str(cost))
            print("learned weight vector: " + str(weights1))
            print("learning rate: " + str(r))
            print(" ")
            r = r * 0.5
            weights0 = weights1


def stochastic_gradient_descent(train_data, iteration=100):
    r = 1
    weights1 = np.asmatrix(np.zeros(7))
    weights0 = np.asmatrix(np.zeros(7))
    train_dataX = np.asmatrix(train_data.iloc[:, :7].values)
    train_dataY = np.asmatrix(train_data.iloc[:, 7:].values)
    print(train_data.shape[0])
    while True:
        for t in range(iteration):
            print("at step" + str(t))
            i = random.randint(0,train_data.shape[0]-1)
            wT = np.transpose(weights0)
            wTx = train_dataX[i] * wT
            error = train_dataY[i] - wTx
            gradient = error * train_dataX[i]
            weights1 = weights0 + r * gradient

            _wT = np.transpose(weights1)
            _wTx = train_dataX * _wT
            _error = train_dataY - _wTx
            cost = np.sum(np.square(error)) * 0.5
            if np.linalg.norm(weights1 - weights0) < 0.000001:
                print("cost: " + str(cost))
                print("Final learned weight vector: " + str(weights1))
                print("Final learning rate: " + str(r))
                return
            print("cost: " + str(cost))
            print("learned weight vector: " + str(weights1))
            print("learning rate: " + str(r))
            print(" ")
            r = r * 0.5
            weights0 = weights1


def optimal_weight_vector(train_data):
    train_dataX = np.asmatrix(train_data.iloc[:, :7].values)
    train_dataY = np.asmatrix(train_data.iloc[:, 7:].values)
    XXT = train_dataX * np.transpose(train_dataX)
    XXT_inverse = np.linalg.inv(XXT)
    temp = XXT_inverse * train_dataX
    optimal_weight = np.transpose(temp) * train_dataY
    print("optimal weight vector = " + str(np.transpose(optimal_weight)))
    return


if __name__ == '__main__':
    train_data = pd.read_csv("concrete/train.csv", header=None)
    train_data.columns = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'y']
    print("RUNNING BATCH GRADIENT DESCENT")
    batch_gradient_descent(train_data)
    print(" ")
    print("RUNNING STOCHASTIC GRADIENT DESCENT")
    stochastic_gradient_descent(train_data)
    print(" ")
    print("OPTIMAL WEIGHT VECTOR")
    optimal_weight_vector(train_data)



