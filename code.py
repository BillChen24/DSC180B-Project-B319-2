import os
import tarfile
import shutil
import hashlib
import glob
import random
import pickle
from datetime import datetime
from typing import *
from numba import jit
import requests
from joblib import Parallel, delayed
from PIL import Image, ImageOps
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

def histOfAnimal(picList, label):
    mapped = list(zip(picList, label))
    listA = [i for i in mapped if i[1] != 'Unknown']
    #listN = [i for i in mapped if i[1] == 'no']
    #list_im = ["images/1a.jpg", "images/3a.jpg"]
    imgs = [ Image.open(i) for i in listA ]
    #imgsN = [ Image.open(i) for i in listN ]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save('comba.jpg')
    imageObj = cv2.imread('comba.jpg')

# Get RGB data from image
    blue_color = cv2.calcHist([imageObj], [0], None, [256], [0, 256])
    red_color = cv2.calcHist([imageObj], [1], None, [256], [0, 256])
    green_color = cv2.calcHist([imageObj], [2], None, [256], [0, 256])

