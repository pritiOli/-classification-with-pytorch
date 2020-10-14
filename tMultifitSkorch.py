
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
import scipy.io
from matplotlib.pyplot import figure as fig

file_path='./datasets/multi-class'
EPOCHS = 1
BATCH_SIZE = 100
LEARNING_RATE = 0.001

test_images= scipy.io.loadmat(os.path.join(file_path,'test_images.mat'))['test_images']
test_labels= scipy.io.loadmat(os.path.join(file_path,'test_labels.mat'))['test_labels'][0]
train_images= scipy.io.loadmat(os.path.join(file_path,'train_images.mat'))['train_images']
train_labels= scipy.io.loadmat(os.path.join(file_path,'train_labels.mat'))['train_labels'][0]

class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_data = trainData(torch.FloatTensor(train_images),torch.FloatTensor(train_labels))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)



class multiClassClassification(nn.Module):
    def __init__(self,num_units1=75,num_units2=15):
        super(multiClassClassification, self).__init__()
        # Number of input features is 10.
        self.hiddenLayer1 = nn.Linear(784, num_units1)
        self.hiddenLayer2 = nn.Linear(num_units1, num_units2)
        self.outputLayer = nn.Linear(num_units2, 10)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.hiddenLayer1(x)
        x = self.relu(x)
        x = self.hiddenLayer2(x)
        x = self.relu(x)
        x = self.outputLayer(x)
        x = self.softmax(x)
        return x



net = NeuralNetClassifier(
    multiClassClassification,
    criterion =  nn.CrossEntropyLoss(),
    optimizer =optim.SGD,
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)


# deactivate skorch-internal train-valid split and verbose logging
net.set_params(train_split=False, verbose=0)
params = {
    'lr': [0.01],
    'max_epochs': [10],
    'module__num_units1': [50,75,100],
    'module__num_units2': [10,15,20],
}

params = {
    'lr': [0.01],
    'max_epochs': [10],
    'module__num_units1': [50],
    'module__num_units2': [10],

}
gs = GridSearchCV(net, params, refit=False, cv=5, scoring='accuracy', verbose=2)



for X, y in train_loader:
    y = y.view(BATCH_SIZE, 1)
    print(y)
    break
    # print(X.shape)
    # print(y.shape)
    gs.fit(X, y)
    print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))



