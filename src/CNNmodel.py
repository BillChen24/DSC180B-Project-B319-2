from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.background import *
from src.rgb_visualization import *
import torch

import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import math
from tqdm import tqdm
from copy import deepcopy

ROOT_STATS_DIR = './experiment_data'

import copy


#Model
class CustomCNN(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup. 
    This will serve as the encoder for our Image Captioning problem.
    '''
    def __init__(self, outputs):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments
        
        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        super(CustomCNN, self).__init__()
        #Check
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print('To Cuda')
            
        #Initialize Layers
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=11,stride=4)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding=2)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.pool21 = nn.MaxPool2d(kernel_size = 3,stride= 2)
        self.pool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_features=128,out_features=1024)
        self.fc2 = nn.Linear(in_features=1024,out_features=1024)
        self.fc3 = nn.Linear(in_features=1024,out_features=outputs)
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.batch_norm2 = nn.BatchNorm2d(num_features=128)
        self.batch_norm3 = nn.BatchNorm2d(num_features=256)
        self.batch_norm4 = nn.BatchNorm2d(num_features=256)
        self.batch_norm5 = nn.BatchNorm2d(num_features=128)

    def forward(self, x):
        '''
        Forward calculate through layers 
        
        INPUT:
            x: Input to the CNN
            dim of x: batchsize x 3 x height x width 
        
        OUTPUT:
            output label prediction: batchsize x class
            
        '''
        # -> bs, 3, 448, 448
        #x= x.to(self.device)
        x = self.pool1(F.relu(self.batch_norm1(self.conv1(x)))) 
        x = self.pool2(F.relu(self.batch_norm2(self.conv2(x))))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool21(F.relu(self.batch_norm5(self.conv5(x))))
        x = self.pool3(x)
        x = torch.flatten(x,start_dim =1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_oneepoch(data_loader, model, loss_func, dataset= None, optimizer = None, learning_rate = 0.0001, bg_remove = False, normalize = False):
    """
    1 epoch update model with whole dataset (in batches ) once 
    
    INPUT:
    data_loader: modified data loader from iwildcam
    model: model object
    loss func: str, loss function type #Not in use 
    optimizer: None or Adam
    learning_rate: float (defualt 0.0001)
    
    OUTPUT:
    average training Loss of given epoch 
    """
    #batch_size=10
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    #for i in enumerate(tqdm(self.__train_loader)):
    for i,databatch in enumerate(tqdm(data_loader)):    
        if optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(),lr=learning_rate)
            optimizer.zero_grad()
        train_image = databatch[0] #batchsize ,3, 448, 448
        train_label = databatch[1]
        #BG subtract
        if bg_remove == True:
            ...
            #train_datas = [train_data[idx] for idx in range(i[1], i[1]+batch_size)]
            #print(len(train_datas))
            #print(train_datas[0])
            #mean_bgs = torch.load('data/Background/mean_background.json')
            #train_image = torch.stack([remove_background(d, mean_bgs, alpha = 0) for d in train_datas])
        #Normalization
        if normalize == True:
            train_image = torch.stack([normalZ(d) for d in train_image])
        #train_image=... #batchsize ,3, 448, 448
        #print(train_image.shape)
        train_label = torch.clip(train_label, min=0, max = 499)
        train_image = train_image.to(model.device)
        train_label = train_label.to(model.device)
       
        batch_size = train_image.shape[0]
        output = model.forward(x=train_image)
        #print(output.shape) #batchsize x class number

        loss = criterion(output,train_label)
        total_loss += loss.item()/batch_size
        # If trainig: 
        loss.backward() #update
        train_image.detach() #TO help free memory
        train_label.detach()
        if optimizer is not None:
            optimizer.step()
    return total_loss/i

def validate_oneepoch(data_loader, model, loss_func, optimizer = None, learning_rate = 0.0001):
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for i in enumerate(tqdm(data_loader)):        
        if optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        optimizer.zero_grad()
        train_image = ... #i[1][0] #batchsize ,3, 256,256
        train_image.to(self.device)
        ...    
        batch_size = train_image.shape[0]
        output = model.forward(x=train_image,captions=copy.deepcopy(train_captions[:,:-1]),batch_size=batch_size,teacher_forcing=True)
       
        loss = criterion(output,train_label)
        total_loss += loss.item()
        # If trainig: 
        loss.backward()
        optimizer.step()
    return total_loss/len(self.__train_loader)

def train_model(total_epoch):
    print("start running")
    #total_epoch = 100
    start_epoch =0 #self.__current_epoch
    #patience_count = 0
    min_loss = 100
    train_losses = []
    
    for epoch in range(start_epoch,total_epoch): 
        print(f'Epoch {epoch + 1}')
        print('--------')
        start_time = datetime.now()
        #self.__current_epoch = epoch
        print('Training...')
        print('-----------')
        train_loss = train_oneepoch(gen, model, 'entropy', optimizer = None, learning_rate = 0.0001) #self.__train()
        train_losses.append(train_loss)
        print('Validating...')
        print('-------------')
        val_loss = validate_oneepoch(gen, model, 'entropy', optimizer = None, learning_rate = 0.0001) #self.__train()

        # save best model
        if val_loss < min_loss:
            min_loss = val_loss
            self.__best_model = "best_model.pt"
            model_dict = self.__model.state_dict()
            state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
            torch.save(state_dict, self.__best_model)

        # early stop if model starts overfitting
        if self.__early_stop:
            if epoch > 0 and val_loss > self.__val_losses[epoch - 1]:
                patience_count += 1
            if patience_count >= self.__patience:
                print('\nEarly stopping!')
                #self.__record_stats(train_loss, val_loss)
                break

       #Save Model
        ...