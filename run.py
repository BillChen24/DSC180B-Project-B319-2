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
from src.normalize import *
from src.iwildcam_dataset import *


def main(trail = 'main', batchsize=64, epoch = 30, bg_remove = False, normalize = False):
    """
    trails: main / test
    Train: trian new model / read exist model
    experiment name: BG sub, OT
    """
   
    #initialize model
    epoch = epoch
    model=CustomCNN(2) #input: Final Class Labels
    model.to(model.device)
    
    #Keep Record
    mean_losses=[]
    mean_losses_val=[]
    accuracy=[]
    accuracy_val=[]
    best_val_acc = 0
    min_val_loss = 1000
    
    #Get Data Loader Based on Trail
    if trail == 'main':
        #dataset = get_dataset(dataset="iwildcam", download=False) #size 203029
        dataset=IWildCamDataset_custom(root_dir='data', download=False, split_scheme='official')
        train_data = dataset.get_subset(
            "train", #'train'
            transform=transforms.Compose(
                [transforms.Resize((448, 448)), transforms.ToTensor()]
            ),)
        dataset=train_data
        #train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=False)
        print('Train Data')
    elif trail == 'test':
        img,ids,info=get_imgs("data/raw/sample/", "data/raw/metadata_test.csv")
        dataset = [img,ids,info]
        #gen=data_test_generator(img, ids, 32)
        print('Test Trail Data')
        
    #Train Model For loop train model for given eqpoch
    for ep in range(epoch):
        gen=get_data_generator(trail, dataset, batchsize = batchsize, shuffle=True)
        epoch_loss=train_oneepoch(gen, model, 'entropy', optimizer ='Adam', learning_rate = 0.0001, bg_remove = bg_remove,normalize = normalize)
        print(epoch_loss)
        mean_losses+=[epoch_loss]
        
        if epoch_loss < min_val_loss:
            min_val_loss = epoch_loss
            model_dict = model.state_dict()
    #test and record prediction
    gen=get_data_generator(trail, dataset, batchsize = batchsize, shuffle=True)
    acc, labels, pred = test_model(gen, model)
    test_record(acc, labels, pred,path = 'result/')
    print('check result')
            
    
    #Display Output
    #Save the best model
    mode_savepath = f"result/best_model_{str(datetime.today())[:10].replace(' ','_')}.pt"
    torch.save(model_dict, mode_savepath)
    print(f'saved the final model to {mode_savepath}')
    return [mean_losses, mean_losses_val, accuracy, accuracy_val]

def Alex(trail = 'alex', batchsize=64, epoch = 10, bg_remove = False, normalize = False):
    """
    trails: main / test
    Train: trian new model / read exist model
    experiment name: BG sub, OT
    """
   
    #initialize model
    epoch = epoch
    model=TransferAlex(50) #input: Final Class Labels
    model.model.to(model.device)
    
    #Keep Record
    mean_losses=[]
    mean_losses_val=[]
    accuracy=[]
    accuracy_val=[]
    best_val_acc = 0
    min_val_loss = 1000
    
    #Get Data Loader Based on Trail
    dataset=IWildCamDataset_custom(root_dir='data', download=False, split_scheme='official')
    train_data = dataset.get_subset(
        "train", #'train'
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),)
    dataset=train_data

        
    #Train Model For loop train model for given eqpoch
    for ep in range(epoch):
        gen=get_data_generator(trail, dataset, batchsize = batchsize, shuffle=True)
        epoch_loss=train_oneepoch_alex(gen, model, 'entropy', optimizer ='SGD', learning_rate = 0.001, bg_remove = bg_remove,normalize = normalize)
        print(epoch_loss)
        mean_losses+=[epoch_loss]

        
        if epoch_loss < min_val_loss:
            min_val_loss = epoch_loss
            model_dict = model.model.state_dict()
            #Save the best model
            mode_savepath = f"result/best_model_{str(datetime.today())[:10].replace(' ','_')}.pt"
            torch.save(model_dict, mode_savepath)
            #state_dict = {'model': model_dict} #'optimizer': self.__optimizer.state_dict()}
            
    #Display Output
    return [mean_losses, mean_losses_val, accuracy, accuracy_val]

def test_record(accs, labels, predictions,path = 'result/'):
    store_path = path+f"prediction_{str(datetime.today())[:19].replace(' ','_')}.txt"
    #Test and record model performance:
    with open(store_path, 'w') as f:
        # Write 5 lines of text to the file
        f.write('-------------Accuracy---------------\n')
        f.write(f'{accs}\n')
        f.write('-------------labels---------------\n')
        f.write(f'{labels}\n')
        f.write('-------------predictions---------------\n')
        f.write(f'{predictions}\n')

def get_imgs(test_data_path, test_meta_path):
    
    # The path to the folder
    folder_path = test_data_path
    # Get a list of all the filenames in the folder
    filenames_exist = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    #Create meta data for test sample
    #metadata = pd.read_csv("data/raw/metadata.csv")
    #metadata_test =metadata[metadata['filename'].apply(lambda x: x in filenames_exist)]
    #metadata_test.iloc[:,1:].to_csv(test_meta_path)
    
    metadata_test=pd.read_csv(test_meta_path, index_col = 0).reset_index()
    filenames = metadata_test['filename'].values
    category_ids=metadata_test['category_id'].values
    images=[]
    labels=[]
    info=[]
    for i, f in enumerate(filenames):
        try:
            img=plt.imread("data/raw/sample/"+f) #read the image
            img=np.transpose(img, (2, 0, 1)) #reshape to 3xheight x width
        except:
            print(f)
        else:
            images+=[img[:,:448,:448]] #crop to 3x448x448
            labels+=[category_ids[i]]
            info+=[metadata_test.iloc[i]['location']]
        if i==200:
            break
    images=torch.from_numpy(np.array(images)).float()
    labels=torch.from_numpy(np.array(labels))#.float()
    infos=torch.from_numpy(np.array(info))
    print(images.shape,labels.shape)
    
    return images, labels,infos

def data_test_generator(dataset, batchsize):
    img, labels,info=dataset
    for i in range(0, len(img), batchsize):
        if (i + batchsize)<len(img):
            yield img[i:i + batchsize], labels[i:i + batchsize],info[i:i + batchsize]
            
def get_data_generator(trail, dataset, batchsize = 64, shuffle=False):
    if trail == 'main':
        gen = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
        #print('Train Data')
    if trail == 'alex':
        gen = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
        #print('Train Data')
    elif trail == 'test':
        gen=data_test_generator(dataset, batchsize)
        #print('Test Trail Data')
    return gen

def draw_loss_plot(results, save_path='result/'):
    for i in results:
        plt.plot(i)
    plt.xlabel('epoch count')
    plt.ylabel('loss & Accuracy')
    plt.title('loss & Accuracy over epochs')
    save_file_path = save_path+f"loss_plot_{str(datetime.today())[:19].replace(' ','_')}.png"
    plt.savefig(save_file_path)
    print(f'save the loss image to {save_file_path}')

if __name__ == '__main__': #if run from command line
    #Prepare path to save result
    path = 'result/'
    #Check and Create dir for result
    if not os.path.exists(path):
        os.mkdir(path)
    targets = sys.argv[1:]
    if len(targets)>1:
        epoch = int(targets[1]);print(epoch)
    else:
        epoch =10
    if targets[0] =='main':
        #Main trail
    #if len(targets)>2:
        results=main('main',batchsize=64,epoch = epoch,bg_remove = True, normalize = True)
    if targets[0] =='test':
        #Test Trail
        results=main('test',batchsize=32,epoch = epoch)
    if targets[0] =='alex':
        #alexnet Trail
        results=Alex('alex',batchsize=32,epoch = epoch,bg_remove = True)
    #NOTE: Results: = [mean_losses, mean_losses_val, accuracy, accuracy_val]
    
    #To Save Result:
    if os.path.exists(path):
        draw_loss_plot(results, save_path='result/')
