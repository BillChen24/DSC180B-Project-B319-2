import random

def get_locations_index(metadata):
    return metadata.groupby(["location_remapped", 'time']).index.apply(list)

def get_location_index(locations_index, location_num):
    return locations_index[location_num].to_dict()

def get_location_data(dataset, location_num, locations_index, time='day'):
    try:
        location_index = get_location_index(locations_index, location_num)[time]
        return [dataset[i] for i in location_index]
    except:
        print("There are no " + time + f" images at location {location_num}.")

def get_location_tensor(dataset, location_num, locations_index, time='day'):
    try:
        location_index = get_location_index(locations_index, location_num)[time]
        return [dataset[i][0] for i in location_index]
    except:
        print("There are no " + time + f" images at location {location_num}.")

def get_location_y(dataset, location_num, locations_index, time='day'):
    try:
        location_index = get_location_index(locations_index, location_num)[time]
        return [dataset[i][1].item() for i in location_index]
    except:
        print("There are no " + time + f" images at location {location_num}.")

def get_location_metadata(dataset, location_num, locations_index, time='day'):
    try:
        location_index = get_location_index(locations_index, location_num)[time]
        return [dataset[i][2] for i in location_index]
    except:
        print("There are no " + time + f" images at location {location_num}.")