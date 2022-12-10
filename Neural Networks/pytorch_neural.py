import pandas as pd
from torch import nn
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F


class classfier(nn.Module):
    def __init__(self, depth, width, activation):
        super().__init__()
        if activation == "RELU":
            self.model = nn.Sequential()
            for k in range(depth - 1):
                self.model.append(nn.Linear(width, width))
            self.model.append(nn.Linear(width, 1))
        else:
            self.model = nn.Sequential()
            for k in range(depth - 1):
                self.model.append(nn.Linear(width, width))
            self.model.append(nn.Linear(width, 1))

    def forward(self, x):
        x = F.relu(self.model(x))
        return self.model(x)


if __name__ == '__main__':
    depth = [3, 6, 9]
    width = [5, 10, 25, 50, 100]
    train_data = pd.read_csv("bank-note/train.csv")
    test_data = pd.read_csv("bank-note/test.csv")
    train_data.columns = ['Variance', 'skewness', 'curtosis', 'entropy', 'y']
    test_data.columns = ['Variance', 'skewness', 'curtosis', 'entropy', 'y']
    train_data["y"] = train_data["y"].replace([0], -1)
    test_data["y"] = test_data["y"].replace([0], -1)
    train_dataX = np.asmatrix(train_data.iloc[:, :4].values)
    train_dataY = np.asmatrix(train_data.iloc[:, 4:].values)
    for i in range(len(depth)):
        for j in range(len(width)):
            clf = classfier(depth[i], width[j], "RELU")
            optimizer = Adam(clf.parameters(), lr=1e-3)
            loss = nn.CrossEntropyLoss()
            random = np.random.permutation(train_data.shape[0] - 1)
            for i in range(train_data.shape[0] - 1):
                random_i = random[i] - 1
                y_hat = clf(train_dataX[random_i])
                error = loss(y_hat, train_dataY[random_i])
                optimizer.zero_grad()
                error.backward()
                optimizer.step()
                print("For depth: " + depth[i] + " width: " + width[j] + " Error = " + error.item())
