# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:21:24 2021

@author: z.rao
"""


# importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score
import seaborn as sns

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py
# from plot3D import *

with h5py.File("Data_remove_0.03_x_0.2_0.8_z_0.03_0.04/data_16.h5", "r") as hf:    

    # Split the data into training/test features/targets
    X_train = hf["X_train_data_3"][:]
    X_train = np.round(X_train)
    targets_train = hf["y_train_data_3"][:]
    targets_train = np.round(targets_train)
    targets_train = np.int64(targets_train).flatten()
    
    X_test = hf["X_test_data_3"][:] 
    X_test = np.round(X_test)
    targets_test = hf["y_test_data_3"][:]
    targets_test = np.round(targets_test)
    targets_test = np.int64(targets_test).flatten()

    
train_x = torch.from_numpy(X_train).float()
train_y = torch.from_numpy(targets_train)


test_x = torch.from_numpy(X_test).float()
test_y = torch.from_numpy(targets_test)


batch_size = 100 #We pick beforehand a batch_size that we will use for the training


# Pytorch train and test sets
train = torch.utils.data.TensorDataset(train_x,train_y)
test = torch.utils.data.TensorDataset(test_x,test_y)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

n_cubic=16
conv=3

def change(predicted):
    for i in range(len(predicted)):
        if predicted[i] >= 0.5:
            predicted[i] = 1
        else:
            predicted[i] = 0
    return predicted

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(2, 16)
        self.conv_layer2 = self._conv_layer_set(16, 32)
        self.conv_layer3 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(10**3*64, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        # nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out

#Definition of hyperparameters
# n_iters = 4500
# num_epochs = n_iters / (len(train_x) / batch_size)
num_epochs = 20

# Create CNN
model = CNNModel()
#model.cuda()
# print(model)

# Cross Entropy Loss 
# error = nn.CrossEntropyLoss()
error = nn.BCEWithLogitsLoss()
# error = nn.MSELoss()

# SGD Optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
train_ls, test_ls = [], []
for epoch in tqdm(range(num_epochs)):
    for i, (images, labels) in enumerate(train_loader):
        

        train = images.view(100,2,n_cubic,n_cubic,n_cubic)
        # labels = Variable(labels)
        
        # labels = torch.flatten(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        outputs = torch.flatten(outputs)
        # print(outputs)
        
        # Calculate softmax and ross entropy loss
        labels=labels.float()
        # print(labels)
        loss = error(outputs, labels)
        # print(loss)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        
        count += 1
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                 
                 test = images.view(batch_size,2,n_cubic,n_cubic,n_cubic)
                 # Forward propagation
                 outputs = model(test)
                 outputs = outputs.view(100)
                 m = nn.Sigmoid()
                 # Get predictions from the maximum value
                 predicted = m(outputs)
                 predicted = predicted.detach().numpy()
                 predicted = change(predicted)
                 # Total number of labels
                 total += len(labels)
                 correct += (predicted == labels.numpy()).sum()
             
             
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        # if count % 500 == 0:
        #     # Print Loss
        #     print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
    train_x = train_x.view(3200,2,n_cubic,n_cubic,n_cubic)
    test_x = test_x.view(800,2,n_cubic,n_cubic,n_cubic)
    train_y=train_y.float()
    test_y=test_y.float()
    train_ls.append(error(model(train_x).view(-1, 1), train_y.view(-1, 1)).item())
    test_ls.append(error(model(test_x).view(-1, 1), test_y.view(-1, 1)).item())
plt.close('all')
sns.set(style='darkgrid')
print ("plot curves")
plt.figure()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(range(1, num_epochs+1),train_ls, linewidth = 5)
plt.plot(range(1, num_epochs+1),  test_ls, linewidth = 5, linestyle=':')
plt.legend(['train loss', 'test loss'])
plt.text(1500, 0.8, 'Loss=%.4f' % test_ls[-1], fontdict={'size': 20, 'color':  'red'})
# plt.show()
plt.savefig('Figures/Remove_0.03_x_0.2_0.8_z_0.03_0.04/16_16_supervised/Supervised_data_3.png', format='png', dpi=300)
torch.save(model.state_dict(), 'Figures/Remove_0.03_x_0.2_0.8_z_0.03_0.04/16_16_supervised/Supervised_data_3.pt')
d={'epoch': range(1, num_epochs+1), 'train_loss':train_ls, 'test_ls': test_ls}
data=pd.DataFrame(data=d)
data.to_csv('Figures/Remove_0.03_x_0.2_0.8_z_0.03_0.04/16_16_supervised/Supervised_data_3.csv')
plt.figure()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(iteration_list,accuracy_list, linewidth = 5)
plt.savefig('Figures/Remove_0.03_x_0.2_0.8_z_0.03_0.04/16_16_supervised/accuracy_data_3.png', format='png', dpi=300)
print ("=== train end ===")