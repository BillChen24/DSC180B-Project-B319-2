import torch
import numpy as np
import random

def get_mean(tensors):
    mean_tensor = torch.mean(torch.stack([t.flatten() for t in tensors]), 0)
    return mean_tensor.reshape(3, 448, 448)

def get_median(tensors):
    median_tensor = torch.median(torch.stack([t.flatten() for t in tensors]), 0).values
    return median_tensor.reshape(3, 448, 448)

def get_location_mean(dataset, location_index, sample = 200):
    """
    get background of a certain location (using mean distribution)
    """
    if 'day' in location_index:
        day_index = location_index['day']
        day_index = random.choices(day_index, k = min(sample, len(day_index)))
        day_mean = get_mean([dataset[i][0] for i in day_index])
    else:
        day_mean = None
    
    if 'night' in location_index:
        night_index = location_index['night']
        night_index = random.choices(night_index, k = min(sample, len(night_index)))
        night_mean = get_mean([dataset[i][0] for i in night_index])
    else:
        night_mean = None
    
    return {"day": day_mean, "night": night_mean}


def get_location_median(dataset, location_index, sample = 200):
    """
    get background of a certain location (using median distribution)
    """
    if 'day' in location_index:
        day_index = location_index['day']
        day_index = random.choices(day_index, k = min(sample, len(day_index)))
        day_median = get_median([dataset[i][0] for i in day_index])
    else:
        day_median = None
    
    if 'night' in location_index:
        night_index = location_index['night']
        night_index = random.choices(night_index, k = min(sample, len(night_index)))
        night_median = get_median([dataset[i][0] for i in night_index])
    else:
        night_median = None
    
    return {"day": day_median, "night": night_median}

def find_background(bg_dict,data = None, meta_array = None):
    if meta_array is None:
        meta_array = data[2]
    location = meta_array[0].item()
    hour = meta_array[5].item()
    if hour > 6 and hour < 18:
        time = 'day'
    else:
        time = 'night'
    return bg_dict[location][time]

def getBinary(subtracted, alpha = 2):
    tnorm = torch.norm(subtracted.reshape(3,448*448), dim = 0)
    norm_mean = tnorm.mean().item()
    norm_std = tnorm.std().item()
    threshold = norm_mean + alpha * norm_std
    M = tnorm.apply_(lambda x: 1 if x >= threshold else 0).reshape(448,448)
    M = torch.stack([M,M,M])
    return M

def getMask(original, binary):
    return torch.mul(original, binary)

def remove_background(data, bg_dict, mask = True, out_binary = False, alpha = 2):
    tensor = data[0]
    bg = find_background(bg_dict, data = data)
    subtracted = tensor - bg
    if mask:
        binary = getBinary(subtracted, alpha)
        if out_binary:
            return binary
        out = getMask(tensor, binary)
    else:
        out = abs(subtracted)
    
    return out
