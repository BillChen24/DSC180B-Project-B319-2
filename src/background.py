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

def find_background(data, bg_dict):
    meta_array = data[2]
    location = meta_array[0].item()
    hour = meta_array[5].item()
    if hour > 6 & hour < 18:
        time = 'day'
    else:
        time = 'night'
    return bg_dict[location][time]
 
def remove_background(data, bg_dict, alpha = 1):
    tensor = data[0]
    bg = find_background(data, bg_dict)
    out = abs(tensor - bg)
    t_mean = torch.mean(out).item()
    return out.apply_(lambda x: 0 if x < alpha * t_mean else x)
