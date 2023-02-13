from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from datetime import datetime
from tqdm import tqdm
import os
import sys
import copy
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


#ROOT_STATS_DIR = './experiment_data'

#IMport self defined py
from src.background import *
from src.rgb_visualization import *
from src.location_group import *
from src.CNNmodel import *


def main(trail = 'main', batchsize=64):
    """
    trails: main / test
    Train: trian new model / read exist model
    experiment name: BG sub, OT
    """
   
    #initialize model
    epoch = 20
    model=CustomCNN(500) #input: Final Class Labels
    model.to(model.device)
    
    #Get Data Loader Based on Trail
    if trail == 'main':
        dataset = get_dataset(dataset="iwildcam", download=False) #size 203029
        train_data = dataset.get_subset(
            "id_val",
            transform=transforms.Compose(
                [transforms.Resize((448, 448)), transforms.ToTensor()]
            ),)
        dataset=train_data
        #train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=False)
        print('Train Data')
    elif trail == 'test':
        img,ids=get_imgs("data/raw/sample/", "data/raw/metadata_test.csv")
        dataset = [img,ids]
        #gen=data_test_generator(img, ids, 32)
        print('Test Trail Data')
        
    #Train Model For loop train model for given eqpoch
    for ep in range(epoch):
        gen=get_data_generator(trail, dataset, batchsize = batchsize, shuffle=False)
        r=train_oneepoch(gen, model, 'entropy', optimizer ='Adam', learning_rate = 0.00001, bg_remove = True)
        print(r) #r: average loss over a epoch
        #validate_oneepoch(data_loader, model, loss_func, optimizer = None, learning_rate = 0.0001)
    
    #Display Output

def get_imgs(test_data_path, test_meta_path):
    
    # The path to the folder
    folder_path = test_data_path
    
    # Get a list of all the filenames in the folder
    filenames_exist = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    #metadata = pd.read_csv("data/raw/metadata.csv")
    #metadata_test =metadata[metadata['filename'].apply(lambda x: x in filenames_exist)]
    #metadata_test.iloc[:,1:].to_csv(test_meta_path)
    metadata_test=pd.read_csv(test_meta_path, index_col = 0).reset_index()
    filenames = metadata_test['filename'].values
    category_ids=metadata_test['category_id'].values
    images=[]
    labels=[]
    for i, f in enumerate(filenames):
        try:
            img=plt.imread("data/raw/sample/"+f)
            img=np.transpose(img, (2, 0, 1))
        except:
            print(f)
        else:
            images+=[img[:,:448,:448]]
            labels+=[category_ids[i]]
        if i==200:
            break
    images=torch.from_numpy(np.array(images)).float()
    labels=torch.from_numpy(np.array(labels))#.float()
    print(images.shape,labels.shape)
    
    return images, labels

def data_test_generator(img, labels, batchsize):
    for i in range(0, len(img), batchsize):
        if i + batchsize<len(img):
            yield img[i:i + batchsize], labels[i:i + batchsize]
            
def get_data_generator(trail, dataset, batchsize = 32, shuffle=False):
    if trail == 'main':
        gen = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
        #print('Train Data')
    elif trail == 'test':
        gen=data_test_generator(dataset[0], dataset[1], batchsize)
        #print('Test Trail Data')
    return gen

if __name__ == '__main__': #if run from command line
    targets = sys.argv[1:]
    if targets[0] =='main':
        main('main',batchsize=64)
    if targets[0] =='test':
        main('test',batchsize=32)
    #main(targets)