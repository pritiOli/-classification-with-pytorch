
"""
Created on Sat Oct 2 11:52:18 2020
@author: Priti
tutorial : https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
tips: Classes must span between 0 to N-1( 22 ). I think your classes span from 1 to 23.
Change batch_y to span between 0 to 22, and keep your output neurons to 23.
I suspect this is the issue.!

https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab

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

import scipy.io

file_path='./datasets/multi-class'
EPOCHS = 10
BATCH_SIZE = 1
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


class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data


    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)

train_data = trainData(torch.FloatTensor(train_images),torch.LongTensor(train_labels))
test_data = testData(torch.FloatTensor(test_images))

y_test = test_labels
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class multiClassClassification(nn.Module):
    def __init__(self):
        super(multiClassClassification, self).__init__()
        # Number of input features is 10.
        self.hiddenLayer1 = nn.Linear(784, 75)
        self.hiddenLayer2 = nn.Linear(75, 10)
        self.outputLayer = nn.Linear(10, 10)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.hiddenLayer1(x)
        x = self.relu(x)
        x = self.hiddenLayer2(x)
        x = self.relu(x)
        x = self.outputLayer(x)
        # x = self.softmax(x)
        return x


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc) * 100

    return acc

model = multiClassClassification()
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

model.train()
for e in range(1, EPOCHS + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, label in train_loader:

        X_batch, label = (X_batch).to(device), label.to(device)
        # print("size of x_batch is ")
        # print(X_batch.shape)

        optimizer.zero_grad()
        # print("shape of output")
        output = model(X_batch)


        # print("\n \n \n label {} ".format(label.view(1,1)))
        loss = criterion(output, label)

        acc = multi_acc(output, label.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')


y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

confusion_matrix(y_test, y_pred_list)
print(classification_report(y_test, y_pred_list))

ax=plt.subplot()
sns.heatmap(cf_matrix, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');

plt.savefig('confusion.png')
plt.close()

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_list)
auc = metrics.roc_auc_score(y_test, y_pred_list)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('auc.png')