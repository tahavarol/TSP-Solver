# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import copy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import torchmetrics
import pickle

size=100
num_cores = 64
batch_size = 64*8
ff_dim = 1024
num_epochs = 250


# %%
class Net(nn.Module):
    
    def __init__(self,TSP_size,ff_dim):
      
        self.TSP_size = TSP_size
        self.ff_dim = ff_dim
        self.n_layers = math.ceil(math.log(TSP_size,2))

        super(Net, self).__init__()

        self.encoder = nn.TransformerEncoderLayer(d_model=self.TSP_size, nhead=self.TSP_size//10, batch_first=True, dim_feedforward=self.ff_dim)
        self.transformer = nn.TransformerEncoder(self.encoder,num_layers=self.n_layers)      



    def forward(self,x):
      
        x = F.relu(x)
        
        out1 = self.transformer(x)
        out2 = torch.transpose(out1, 1, 2)
        x1 = F.softmax(out1,1)
        x2 = F.softmax(out2,2)
        x3 = torch.add(x1, x2)
        x4 = F.hardtanh(x3, 0, 1)

        return x4



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def is_symmetric(matrix):
    # Check if the matrix is equal to its transpose
    return torch.allclose(matrix, matrix.t())
    

# %%
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.tsp_size = (data.shape[1]-1)//3
        self.data = data[:,:2*self.tsp_size]
        self.labels = data[:,2*self.tsp_size:3*self.tsp_size]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_label = self.labels[idx]

        if self.transform:
            sample_data,sample_label = self.transform((sample_data,sample_label))
            

        return sample_data, sample_label

class ToTensor(object):
    def __call__(self, data):

        sample_data = data[0]
        sample_label = data[1]
        
        size = sample_data.shape[0]//2
        X = np.column_stack((sample_data[:size], sample_data[size:]))
        dist = euclidean_distances(X,X)
        dist = dist/np.max(dist)
        dist = 1-dist
        np.fill_diagonal(dist, 0)
        dist = torch.tensor(dist, dtype=torch.float32)

        mroutelist = sample_label.tolist()
        mroutelist.append(0)
        route_matrix = np.zeros((size,size))
        
        for i in range(size):
            origin = int(mroutelist[i])
            dest = int(mroutelist[i+1])
            route_matrix[origin,dest] = 1
            route_matrix[dest,origin] = 1
     

        route_matrix = torch.tensor(route_matrix, dtype=torch.float32)

        return dist, route_matrix


# %%
data = np.loadtxt("TrainingData/Raw/{}.csv".format(size), delimiter=',')

train, val = train_test_split(data, random_state=42, test_size=0.05)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%

train_dataset = CustomDataset(train, transform=ToTensor())
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_cores)

val_dataset = CustomDataset(val, transform=ToTensor())
val_data_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=num_cores)

# %%
model = Net(size,ff_dim=ff_dim)
print(count_parameters(model))
model.to(device)

# %%
train_acc = torchmetrics.Accuracy().to(device)
train_precision = torchmetrics.Precision().to(device)
train_recall = torchmetrics.Recall().to(device)
train_F1 = torchmetrics.F1().to(device)

val_acc = torchmetrics.Accuracy().to(device)
val_precision = torchmetrics.Precision().to(device)
val_recall = torchmetrics.Recall().to(device)
val_F1 = torchmetrics.F1().to(device)

# %%
# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam([x for x in model.parameters() if x.requires_grad], lr=0.001)

# Hold the best model
best_loss = np.inf   # init to infinity
best_weights = None
optimizer.zero_grad()

history = {"train_loss":[],
           "train_accuracy":[],
           "train_precision":[],
           "train_recall":[],
           "train_F1":[],
           "validation_loss":[],
           "validation_accuracy":[],
           "validation_precision":[],
           "validation_recall":[],
           "validation_F1":[]}

MODEL_PATH = "best_model_{}.pth".format(size)


for epoch in range(num_epochs):
        train_running_loss = 0.
        train_last_loss = 0.
        train_running_acc = 0.
        train_last_acc = 0.
        train_running_precision = 0.
        train_last_precision = 0.
        train_running_recall = 0.
        train_last_recall = 0.
        train_running_F1 = 0.
        train_last_F1 = 0.        
        for i,batch in enumerate(tqdm(train_data_loader)):
                d, l = batch 
                d, l = d.to(device), l.to(device)
                # forward pass
                y_pred = model(d)
                loss = loss_fn(y_pred, l)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print progress
                train_running_loss += loss.item()
                train_running_acc  += train_acc(y_pred, l.long()).item()
                train_running_precision  += train_precision(y_pred, l.long()).item()
                train_running_recall += train_recall(y_pred, l.long()).item()
                train_running_F1 += train_F1(y_pred, l.long()).item()
                del(d,l)
        train_last_loss = train_running_loss/len(train_data_loader)
        train_last_acc = train_running_acc/len(train_data_loader)
        train_last_precision = train_running_precision/len(train_data_loader)
        train_last_recall = train_running_recall/len(train_data_loader)
        train_last_F1 = train_running_F1/len(train_data_loader)
        print("epoch: ", epoch, "train_loss: ",train_last_loss, "train_acc: ",train_last_acc,"train_F1: ", train_last_F1)
        history["train_loss"].append(train_last_loss)
        history["train_accuracy"].append(train_last_acc)
        history["train_precision"].append(train_last_precision)
        history["train_recall"].append(train_last_recall)
        history["train_F1"].append(train_last_F1)
# evaluate accuracy at end of each epoch
        model.eval()
        val_running_loss = 0.
        val_last_loss = 0.
        val_running_acc = 0.
        val_last_acc = 0.
        val_running_precision = 0.
        val_last_precision = 0.
        val_running_recall = 0.
        val_last_recall = 0.
        val_running_F1 = 0.
        val_last_F1 = 0.       
        for i,batch in enumerate(tqdm(val_data_loader)):
                d, l = batch 
                d, l = d.to(device), l.to(device)
                y_pred = model(d)
                val_loss = loss_fn(y_pred, l)
                val_running_loss += val_loss.item()
                val_running_acc  += val_acc(y_pred, l.long()).item()
                val_running_precision  += val_precision(y_pred, l.long()).item()
                val_running_recall += val_recall(y_pred, l.long()).item()
                val_running_F1 += val_F1(y_pred, l.long()).item()
                del(d,l)
        val_last_loss = val_running_loss/len(val_data_loader)
        val_last_acc = val_running_acc/len(val_data_loader)
        val_last_precision = val_running_precision/len(val_data_loader)
        val_last_recall = val_running_recall/len(val_data_loader)
        val_last_F1 = val_running_F1/len(val_data_loader)
        print("epoch: ", epoch, "val_loss: ",val_last_loss, "val_acc: ",val_last_acc,"val_F1: ", val_last_F1)
        history["validation_loss"].append(val_last_loss)
        history["validation_accuracy"].append(val_last_acc)
        history["validation_precision"].append(val_last_precision)
        history["validation_recall"].append(val_last_recall)
        history["validation_F1"].append(val_last_F1)
        if val_last_loss < best_loss:
                best_loss = val_last_loss
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, MODEL_PATH)

pickle.dump(history, open("training_logs_{}.pkl".format(size), "wb"))
