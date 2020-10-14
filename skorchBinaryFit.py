
"""
Created on Sat Oct 2 11:52:18 2020
@author: Priti
tutorial : https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import numpy as np
import os
from os import listdir
from os.path import isfile,join
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

from torch.utils.tensorboard import SummaryWriter

file_path='./datasets/bi-class'
# writer = SummaryWriter('./binary1')

data_source = listdir(file_path)

EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001

class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def load_data(data_src):
    print("------------{}-------------".format(data_src))
    dataset=np.load(os.path.join(file_path,data_src))
    print("data shape")
    print(dataset["train_X"].shape)
    print(dataset["train_Y"].shape)
    print("--------------------------------------------------")
    train_data = trainData(torch.FloatTensor(dataset["train_X"]),
                           torch.FloatTensor(dataset["train_Y"]))
    return train_data


train_data=load_data(data_source[4])
data = np.load(os.path.join(file_path, data_source[4]))
y_test = data["test_Y"]
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class binaryClassification(nn.Module):
    def __init__(self,num_units=10):
        super(binaryClassification, self).__init__()
        # Number of input features is 10.
        self.hiddenLayer = nn.Linear(13, num_units)
        self.outputLayer = nn.Linear(num_units, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.hiddenLayer(x)
        x = self.relu(x)
        x = self.outputLayer(x)
        # x = self.softmax(x)
        return x



net = NeuralNetClassifier(
    binaryClassification,
    criterion = nn.BCEWithLogitsLoss,
    optimizer =optim.SGD,
    max_epochs=3,
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)


# deactivate skorch-internal train-valid split and verbose logging
net.set_params(train_split=False, verbose=0)
params = {
    'lr': [0.01, 0.05,0.001,0.005],
    'max_epochs': [10, 20,30],
    'module__num_units': [1,2,3,4,5,6,7,8,9,10],
}
gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy', verbose=2)



for X, y in train_loader:
    y = y.view(BATCH_SIZE, 1)

    # print(X.shape)
    # print(y.shape)
    gs.fit(X, y)
    print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))



