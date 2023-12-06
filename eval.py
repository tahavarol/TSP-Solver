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
import time


global ff_dim
ff_dim = 1024


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
        self.lengths = data[:,-1]
        self.labels = data[:,2*self.tsp_size:3*self.tsp_size]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_label = self.labels[idx]
        sample_lengths = self.lengths[idx]

        if self.transform:
            sample_data,sample_label,sample_raw_dist = self.transform((sample_data,sample_label))
            

        return sample_data, sample_label, sample_raw_dist, sample_lengths

class ToTensor(object):
    def __call__(self, data):

        sample_data = data[0]
        sample_label = data[1]
        
        size = sample_data.shape[0]//2
        X = np.column_stack((sample_data[:size], sample_data[size:]))
        dist = euclidean_distances(X,X)
        dist_ = dist/np.max(dist)
        dist_ = 1-dist_
        np.fill_diagonal(dist_, 0)
        dist_ = torch.tensor(dist_, dtype=torch.float32)
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

        return dist_, route_matrix, dist

def greedy_decode_batch(output_matrix, distance_matrix):
    batch_size, num_points, _ = output_matrix.size()

    # Initialize the tour tensor
    best_tour = torch.zeros(batch_size, num_points, dtype=torch.long)

    for i in range(batch_size):
       output_matrix[i].fill_diagonal_(0)


    best_tour_length = torch.full((batch_size,), float('inf'))

    for start_point in range(num_points):
        # Start from the current point for each instance in the batch
        current_points = torch.full((batch_size,), start_point, dtype=torch.long)
        visited_points = torch.zeros(batch_size, num_points, dtype=torch.bool)
        tour = torch.zeros(batch_size, num_points, dtype=torch.long)
        tour[torch.arange(batch_size),0] = current_points

        # Perform greedy decoding for the current starting point
        for step in range(num_points-1):
            visited_points[torch.arange(batch_size), current_points] = True
            probabilities = output_matrix[torch.arange(batch_size), current_points, :]
            probabilities[visited_points] = 0
            # Choose the next point with the highest probability
            next_points = torch.argmax(probabilities, dim=-1)
            tour[:, step+1] = next_points
            
            current_points = next_points

        # Calculate the tour length for each sample in the batch
        tour_lengths = torch.sum(distance_matrix[torch.arange(batch_size).unsqueeze(1), tour, torch.roll(tour, shifts=-1, dims=1)], dim=-1)
        # Update the best tour if the current one has a shorter length
        update_mask = tour_lengths < best_tour_length
        best_tour[update_mask] = tour[update_mask]
        best_tour_length[update_mask] = tour_lengths[update_mask]

    return best_tour, best_tour_length


def greedy_total(output_matrix, normalized_dist, distance_matrix):

    pred_tour, pred_tour_length = greedy_decode_batch(output_matrix, distance_matrix)

    sum_tour, sum_tour_length = greedy_decode_batch(1-normalized_dist + output_matrix, distance_matrix)

    mult_tour, mult_tour_length = greedy_decode_batch((1 - normalized_dist) * output_matrix, distance_matrix)


    all_tour_lengths = torch.stack([pred_tour_length, sum_tour_length, mult_tour_length], dim=-1)
    all_tours = torch.stack([pred_tour, sum_tour, mult_tour], dim=-1)

    _, min_indices = torch.min(all_tour_lengths, dim=-1)
    min_tour_lengths = torch.gather(all_tour_lengths, dim=-1, index=min_indices.unsqueeze(-1)).squeeze(-1)
    min_tours = all_tours[torch.arange(all_tours.size(0)).unsqueeze(1), :, min_indices.unsqueeze(-1)]

    return min_tours.squeeze(1), min_tour_lengths

def calculate_tour_length(distance_matrix, tour):
    
    batch_size, num_points, _ = distance_matrix.size()
    
    # Create a tensor to hold the tour lengths for each batch

    tour_lengths = torch.sum(distance_matrix[torch.arange(batch_size).unsqueeze(1), tour, torch.roll(tour, shifts=-1, dims=1)], dim=-1)


    return tour_lengths

def two_opt_swap(tour, i, j):
    # Perform 2-opt swap between edges (i, i+1) and (j, j+1)
    new_tour = tour.clone()
    new_tour[:, i:j+1] =torch.flip(tour[:, i:j+1], dims=[1])
    return new_tour

def two_opt(distance_matrix, tour):
    batch_size, num_points, _ = distance_matrix.size()

    # Calculate the initial tour length
    initial_tour_length = calculate_tour_length(distance_matrix, tour)

    # Perform 2-opt swaps
    for i in range(1, num_points - 2):
        for j in range(i + 1, num_points):
            # Calculate the new tour length after the 2-opt swap
            new_tour = two_opt_swap(tour, i, j)
            new_tour_length = calculate_tour_length(distance_matrix, new_tour)
            # Identify tours with shorter lengths
            improve_mask = new_tour_length < initial_tour_length
            
            # Update tours and lengths for improved cases
            tour[improve_mask] = new_tour[improve_mask]
            initial_tour_length[improve_mask] = new_tour_length[improve_mask]

    return tour, initial_tour_length


def GetGap(lns,predicted_tours):

    sm = []
    for i1,i2 in zip(lns,predicted_tours):
    
        gap = i2.item()-i1.item()
        sm.append(gap/i1.item())
   
    
    return np.mean(sm)*100
    
    
def GreedyDecoder_(size, type):

    data = np.loadtxt("BenchmarkInstances/{}_{}.csv".format(type, size), delimiter=',')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = CustomDataset(data, transform=ToTensor())

    val_data_loader = DataLoader(val_dataset, batch_size = len(data), shuffle=False, num_workers=1)

    model = Net(size,ff_dim=ff_dim)
    model.to(device)
    model.load_state_dict(torch.load("best_model_{}.pth".format(size), map_location=torch.device('cpu')))
    model.eval()
    start = time.time()
    for i,batch in enumerate(val_data_loader):
        with torch.no_grad():
            preds = model(batch[0].to(device))
            labels = batch[1]
            dist_mat = batch[2]
            lns = batch[3]
            decoded_tour, decoded_length = greedy_total(preds,batch[0],dist_mat)
            two_opt_tour, two_opt_length = two_opt(dist_mat, decoded_tour)

            current_gap = GetGap(lns,two_opt_length)

            tolerance = 100

            while tolerance > 0.0000001:
                two_opt_tour, two_opt_length = two_opt(dist_mat, two_opt_tour)
                new_gap = GetGap(lns,two_opt_length)
                #print(tolerance, new_gap, current_gap)
                tolerance = current_gap-new_gap
                current_gap = new_gap

    end = time.time()

    my_dict = {}

    my_dict["time"] = end-start
    my_dict["mean_gap"] = new_gap
    my_dict["max_gap"] = 100*(torch.max(two_opt_length/lns).item()-1)
    my_dict["min_gap"] = 100*(torch.min(two_opt_length/lns).item()-1)
    my_dict["perf"] = torch.div(two_opt_length,lns).numpy()-1


    return my_dict
   
instance_names = ["G1", "G2_2", "G3_1", "G3_2", "G3_3", "G4", "SG", "US", "UR", "NS", "NR"]
res_list = []
for item in instance_names:
    res_list.append(GreedyDecoder_(20, item))


print(res_list)