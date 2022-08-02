# importing the libraries
from tkinter.messagebox import NO
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


# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
# from plot3D import *
import glob
import pandas as pd
from dataset import DataLoader
import time

# Create CNN Model


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set1(3, 16, 1, 1, 2)
        self.conv_layer2 = self._conv_layer_set1(16, 32, 1, 2, 2)
        self.conv_layer3 = self._conv_layer_set1(32, 32, 1, 1, 1)
        self.conv_layer4 = self._conv_layer_set1(32, 64, 1, 2, 2)
        self.conv_layer5 = self._conv_layer_set1(64, 128, 1, 2, 2)
        self.conv_layer6 = self._conv_layer_set1(128, 128, 1, 1, 1)
        self.conv_layer7 = self._conv_layer_set1(128, 192, 1, 2, 2)
        self.conv_layer8 = self._conv_layer_set1(192, 256, 1, 2, 2)
        self.conv_layer9 = self._conv_layer_set2(256, 256, 1, 1, 1)
        self.conv_layer10 = self._conv_layer_set2(256, 384, 1, 2, 2)
        self.conv_layer11 = self._conv_layer_set2(384, 512, 1, 2, 2)
        self.lstm_layer1 = nn.LSTM(512, 128, 2, batch_first=True, dropout=0.2)
        self.lstm_layer2 = nn.LSTM(128, 32, 2, batch_first=True, dropout=0.2)
        self.lstm_layer3 = nn.LSTM(32, 1, batch_first=True)
        self.leakyRelu = nn.LeakyReLU()
        self.fc = nn.Linear(50, 1)

    
    def _conv_layer_set1(self, in_c, out_c, stride_d, stride_h, stride_w):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(1, 1, 1), stride=(stride_d, stride_h, stride_w)),
        nn.LeakyReLU(),
        nn.BatchNorm3d(out_c)
        )
        return conv_layer
    
    def _conv_layer_set2(self, in_c, out_c, stride_d, stride_h, stride_w):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(1, 1, 1), stride=(stride_d, stride_h, stride_w)),
        nn.LeakyReLU(),
        nn.BatchNorm3d(out_c)
        )
        return conv_layer
    
    # def _lstm_layer_set(self, input_size, hidden_size, dropout):
    #     lstm_layer = nn.Sequential(
    #         nn.LSTM(input_size, hidden_size, 2, batch_first=True, dropout=dropout),
    #         nn.LeakyReLU()
    #     )
    #     return lstm_layer
    
    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        # print(out.shape)
        out = self.conv_layer2(out)
        # print(out.shape)
        out = self.conv_layer3(out)
        # print(out.shape)
        out = self.conv_layer4(out)
        # print(out.shape)
        out = self.conv_layer5(out)
        # print(out.shape)
        out = self.conv_layer6(out)
        # print(out.shape)
        out = self.conv_layer7(out)
        # print(out.shape)
        out = self.conv_layer8(out)
        # print(out.shape)
        out = self.conv_layer9(out)
        # print(out.shape)
        out = self.conv_layer10(out)
        # print(out.shape)
        out = self.conv_layer11(out)
        # print(out.shape)
        out = out.view(out.size(0), 50, 512)
        # print(out.shape)
        out = self.lstm_layer1(out)
        # print(out[0].shape)        
        # out = self.leakyRelu(out[0])
        out = self.lstm_layer2(out[0])
        # print(out[0].shape)        
        # out = self.leakyRelu(out[0])
        out = self.lstm_layer3(out[0])
        # print(out[0].shape)        
        # out = self.leakyRelu(out[0])
        out = out[0].view(out[0].size(0), -1)
        out = self.fc(out)
        
        return out


def get_data_path(frame_paths):
    # target_arr = []
    # num = 0
    
    data_path = []
    for path in frame_paths:
        # print(num)
        # print(path)
        # print('pass1')
        frame_file = f"{path}/frames.npy"
        hr_file = f"{path}/hr.csv"
        
        if glob.glob(frame_file) == []:
            continue
        
        data_path.append(path)
        
    return data_path

def sortNum(text):
    split_arr = text.split('.')
    num = split_arr[0].split('_')[-1]
    return int(num)

def rmse(l1, l2):
    
    return np.sqrt(np.mean((l1-l2)**2))


def mae(l1, l2):

    return np.mean([abs(item1-item2)for item1, item2 in zip(l1, l2)])

def compute_criteria(target_hr_list, predicted_hr_list):
    pearson_per_signal = []
    HR_MAE = mae(np.array(predicted_hr_list), np.array(target_hr_list))
    HR_RMSE = rmse(np.array(predicted_hr_list), np.array(target_hr_list))

    # for (gt_signal, predicted_signal) in zip(target_hr_list, predicted_hr_list):
    #     r, p_value = pearsonr(predicted_signal, gt_signal)
    #     pearson_per_signal.append(r)

    # return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE, "Pearson": np.mean(pearson_per_signal)}
    return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE}
            
            
if __name__ == '__main__':
    
    # print(torch.cuda.is_available)
    
    train_paths = sorted(glob.glob('data/train/*'), key=sortNum)
    
    test_paths = sorted(glob.glob('data/test/*'), key=sortNum)

    batch_size = 8 #We pick beforehand a batch_size that we will use for the training
    
    train_set = DataLoader(train_paths, 'train')
    test_set = DataLoader(test_paths, 'test')

    # data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size = batch_size, shuffle = False)
    
    print('success')

    num_epochs = 30

    # Create CNN
    model = CNNModel()
    model.cuda(2)
    # print(model)

    # Cross Entropy Loss 
    error = nn.MSELoss()

    # SGD Optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pg = optimizer.param_groups[0]

    # CNN model training
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    for epoch in range(num_epochs):
        print(epoch)
        if epoch == 10:
            pg['lr'] = 1e-4
            print(pg, flush=True)
        if epoch == 20:
            pg['lr'] = 1e-5
            print(pg, flush=True)

        for i, (images, labels) in enumerate(train_loader):
            # print(images.shape)
            train = Variable(images)
            labels = Variable(labels)
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(train)
            outputs = outputs.squeeze(1)
            if count % 50 == 0:
                print(i, flush=True)
                print(outputs, flush=True)
                print(labels, flush=True)
            # Calculate loss
            loss = error(outputs, labels)
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            
            count += 1
            if count % 50 == 0:
                model.cuda(0)
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Iterate through test dataset
                test_num = 0
                for images, labels in test_loader:
                    
                    test = Variable(images.view(images.size(0),3,50,128,256))
                    # Forward propagation
                    torch.cuda.synchronize()
                    start = time.time()
                    outputs = model(test)
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - start

                    print(elapsed_time, 'sec.', flush=True)

                    outputs = outputs.squeeze(1)

                    # print(outputs.data)
                    # Get predictions from the maximum value
                    # predicted = torch.max(outputs.data, 1)[1]
                    predicted = outputs
                    
                    # if test_num == 0: 
                    #     print(test_num, flush=True)
                    # Total number of labels
                    total += len(labels)
                    for n in range(len(labels)):
                        # print(n)
                        # if test_num == 0: 
                        print(f"predicted: {predicted[n]} | label: {labels[n]}", flush=True)
                        if int(predicted[n]) == int(labels[n]):
                            correct += 1
                            continue
                        else:
                            continue

                    test_num += 1    
                    # correct += (predicted == labels).sum()
                    
                print("correct", flush=True)
                print(correct, flush=True)
                # print(total, flush=True)
                accuracy = 100 * correct / float(total)
                print("accuracy", flush=True)
                print(accuracy, flush=True)
                
                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                model.cuda(2)
            if count % 500 == 0:
                metrics = compute_criteria(labels, predicted)
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy), flush=True)
                print('loss list', flush=True)
                print(loss_list, flush=True)
                print('accuracy list', flush=True)
                print(accuracy_list, flush=True)