from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.background import *
from src.rgb_visualization import *
from src.normalize import *
from src.iwildcam_dataset import *
from src.id_mapper import *
import torch

import random
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from datetime import datetime
import math
from tqdm import tqdm
from copy import deepcopy

ROOT_STATS_DIR = './experiment_data'

import copy


#Model 1
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
    

def train_oneepoch(data_loader, model, loss_func, dataset= None, optimizer = None, learning_rate = 0.0001, bg_remove = False, binary = False, normalize = False):
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
    
    if optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    for i,databatch in enumerate(tqdm(data_loader)):    
        
        optimizer.zero_grad()
        train_image = databatch[0] #batchsize ,3, 448, 448
        train_label = databatch[1]
        train_meta = databatch[2]
        #BG subtract
        if bg_remove == True:
            median_bgs = torch.load('data/Background/median_background.json')
            bgs = torch.stack([find_background(median_bgs, meta_array = meta) for meta in train_meta])
            subtracted = train_image-bgs
            masks = torch.stack([getBinary(s, alpha = 1) for s in subtracted])
            train_image = torch.mul(train_image, masks)
            if binary == True:
                train_image = masks
            
        #Normalization
        if normalize == True:
            train_image = torch.stack([normalZ(d) for d in train_image])
        train_label = Category_id_order_mapper(catId=train_label, Catorder=None, id2order=True, binary=True) #Map class id to index
        #train_label = Category_id_order_mapper_binary(catId=train_label, Catorder=None, id2order=True)
       
        train_label=torch.tensor(train_label,dtype=torch.long)
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
        if i ==20:
            break
    return total_loss/i

def validation_oneepoch(data_loader, model, loss_func, dataset= None, optimizer = None, learning_rate = 0.0001, bg_remove = False, normalize = False):
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
    
    for i,databatch in enumerate(tqdm(data_loader)):    
#         if optimizer == 'Adam':
#             optimizer = optim.Adam(model.parameters(),lr=learning_rate)
#             optimizer.zero_grad()
        train_image = databatch[0] #batchsize ,3, 448, 448
        train_label = databatch[1]
        train_meta = databatch[2]
        #BG subtract
        if bg_remove == True:
            median_bgs = torch.load('data/Background/median_background.json')
            bgs = torch.stack([find_background(median_bgs, meta_array = meta) for meta in train_meta])
            subtracted = train_image-bgs
            masks = torch.stack([getBinary(s) for s in subtracted])
            train_image = torch.mul(train_image, masks)
            
        #Normalization
        if normalize == True:
            train_image = torch.stack([normalZ(d) for d in train_image])
      
        train_label = Category_id_order_mapper(catId=train_label, Catorder=None, id2order=True) #Map class id to index
        
        train_label=torch.tensor(train_label,dtype=torch.long)
        train_image = train_image.to(model.device)
        train_label = train_label.to(model.device)
       
        batch_size = train_image.shape[0]
        output = model.forward(x=train_image)
        #print(output.shape) #batchsize x class number

        loss = criterion(output,train_label)
        total_loss += loss.item()/batch_size
       
        loss.backward() #update
        train_image.detach() #TO help free memory
        train_label.detach()
       
        if i ==20:
            break
    return total_loss/i


def test_model(data_loader, model, dataset= None,  bg_remove = False, normalize = False):
    """
    test model accuracy
    """
    #batch_size=10
 

    predictions =[]
    outputs=[]
    train_labelsorigin=[]
    train_labels=[]
    #train_loc=[]
    #train_hour=[]
    
    for i,databatch in enumerate(tqdm(data_loader)):    
#         if optimizer == 'Adam':
#             optimizer = optim.Adam(model.parameters(),lr=learning_rate)
#             optimizer.zero_grad()
        train_image = databatch[0] #batchsize ,3, 448, 448
        train_label = databatch[1]
        train_meta = databatch[2]

        #train_loc=train_loc+train_meta[:,0].tolist()
        #train_hour=train_hour+train_meta[:,5].tolist()

        train_labelsorigin+=train_label.tolist()
        train_label = Category_id_order_mapper(catId=train_label, Catorder=None, id2order=True) #Change to mapper
        train_label=torch.tensor(train_label,dtype=torch.long)

        #BG subtract
        if bg_remove == True:
            median_bgs = torch.load('data/Background/median_background.json')
            bgs = torch.stack([find_background(median_bgs, meta_array = meta) for meta in train_meta])
            subtracted = train_image-bgs
            masks = torch.stack([getBinary(s) for s in subtracted])
            train_image = torch.mul(train_image, masks)
            
        #Normalization
        if normalize == True:
            train_image = torch.stack([normalZ(d) for d in train_image])
      
        train_label = Category_id_order_mapper(catId=train_label, Catorder=None, id2order=True) #Map class id to index
        
        train_label=torch.tensor(train_label,dtype=torch.long)
        train_image = train_image.to(model.device)
        train_label = train_label.to(model.device)
       
        batch_size = train_image.shape[0]
        output = model.forward(x=train_image)
        #print(output.shape) #batchsize x class number

        predictions+=((torch.argmax(output,axis=1)==train_label)).to(torch.float).tolist()
        outputs+=(torch.argmax(output,axis=1)).tolist()
        train_labels+=train_label.tolist()

        train_image.detach() #TO help free memory
        train_label.detach()

        print('Overall Accuracy:')
        print(sum(predictions)/len(predictions))
       
    return sum(predictions)/len(predictions)


#Model 2
class TransferAlex():
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
       
        #Check
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print('To Cuda')
            
        #Initialize Layers
        self.model = models.alexnet(pretrained=True)
        for param in self.model.features.parameters():
            param.requires_grad = False
            
        num_classes = outputs
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
        
        for param in self.model.classifier[-1].parameters():
            param.requires_grad = True
        
    
    def forward(self, x):
        '''
        Forward calculate through layers 
        
        INPUT:
            x: Input to the CNN
            dim of x: batchsize x 3 x height x width 
        
        OUTPUT:
            output label prediction: batchsize x class
            
        '''
        # Define the transformation pipeline
        transform = transforms.Compose([
        transforms.Resize(256),   # Resize the input image to 256x256
        transforms.CenterCrop(224),   # Crop the center of the image to 224x224
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        x = transform(x)
        x=self.model(x)
        
        return x
    
def train_oneepoch_alex(data_loader, model, loss_func, dataset= None, optimizer = 'SGD', learning_rate = 0.0001, bg_remove = False, binary = False, normalize = False):
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
    optimizer = torch.optim.SGD(model.model.parameters(), lr=0.001, momentum=0.9)

    for i,databatch in enumerate(tqdm(data_loader)):    
        
        optimizer.zero_grad()
        train_image = databatch[0] #batchsize ,3, 448, 448
        train_label = databatch[1]
        train_meta = databatch[2]
        #BG subtract
        if bg_remove == True:
            median_bgs = torch.load('data/Background/median_background.json')
            bgs = torch.stack([find_background(median_bgs, meta_array = meta) for meta in train_meta])
            subtracted = train_image-bgs
            masks = torch.stack([getBinary(s, alpha = 1) for s in subtracted])
            train_image = torch.mul(train_image, masks)
            if binary == True:
                train_image = masks
            
        #Normalization
        if normalize == True:
            train_image = torch.stack([normalZ(d) for d in train_image])
       
        train_label = Category_id_order_mapper(catId=train_label, Catorder=None, id2order=True, binary=False) #Map class id to index
        #train_label = Category_id_order_mapper_binary(catId=train_label, Catorder=None, id2order=True)
       
        train_label=torch.tensor(train_label,dtype=torch.long)
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
        if i ==100:
            break
    return total_loss/i

    