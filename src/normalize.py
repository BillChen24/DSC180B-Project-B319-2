from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy import stats
import torch
    
def normalMinMax(data,new_min = 0,new_max = 0):
    img = data[0]
    v_min, v_max = img.min(), img.max()
    new_min, new_max = -.25, .25
    normImg = (img - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    return normImg

def normalZ(data):
    normImg = torch.clone(data)
    for i in range(3):
        img = data[i]
        #img = torch.clone(img)
        normImg[i] = stats.zscore(img)#/256
    return normImg
