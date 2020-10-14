
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

file_path='./datasets/bi-class'

data_source = listdir(file_path)

EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.01


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



def load_data(data_src):
    print("------------{}-------------".format(data_src))
    dataset=np.load(os.path.join(file_path,data_src))
    print("data shape")
    print(dataset["train_X"].shape)
    print(dataset["train_Y"].shape)
    print("--------------------------------------------------")
    train_data = trainData(torch.FloatTensor(dataset["train_X"]),
                           torch.FloatTensor(dataset["train_Y"]))
    test_data = testData(torch.FloatTensor(dataset["test_X"]))

    return train_data,test_data


# for dataset in data_source:
#     train_data,test_data=load_data(dataset)
#     train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(dataset=test_data, batch_size=1)


train_data,test_data=load_data(data_source[4])
data = np.load(os.path.join(file_path, data_source[4]))
y_test = data["test_Y"]
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 10.
        self.hiddenLayer = nn.Linear(13, 9)
        self.outputLayer = nn.Linear(9, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.hiddenLayer(x)
        x = self.relu(x)
        x = self.outputLayer(x)
        # x = self.softmax(x)
        return x

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

model = binaryClassification()
print(model)
criterion = nn.BCEWithLogitsLoss()
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

        # print(output.shape)
        # print("size of label is")
        # print(label.shape)
        label=label.view(1,1)

        loss = criterion(output, label)

        acc = binary_acc(output, label.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
cf_matrix=confusion_matrix(y_test, y_pred_list)
print(classification_report(y_test, y_pred_list))
# fig=sns.heatmap(cf_matrix, annot=True)
# pl.title('Confusion matrix of the classifier')
# x= plt.subplot()

ax=plt.subplot()
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.heatmap(cf_matrix, annot=True,cmap=cmap,annot_kws={"size": 12},vmax=400, vmin=0, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');

plt.savefig('./analyses/confusion5.png')
plt.close()

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_list)
auc = metrics.roc_auc_score(y_test, y_pred_list)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('./analyses/auc5.png')