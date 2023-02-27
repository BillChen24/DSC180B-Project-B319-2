"""
We are Not using the entire dataset

For training dataset:
1. select locations with more than 1000 image input
2. For all images in selected location, keep categories that has more than 100 pictures
Expect: 40 locations, 42 categories

Mapper:
For training purpose, the category ids has to be mapper to values from 0 to number of avaible categories -1 
And for testing and analysis purpose, mapping should be both direction
"""

import numpy as np
folderpath = ''

#Find Locations with more than 1000 images
def filter_trainingset(meta, save=False):
    """
    IN: csv of meta data of iwildcams
    OUT: arrays contraining eligable location ids and category ids
    """
    g=meta.groupby(['split','location']).count().reset_index()
    locations=g[(g['split']=='train')&(g['y']>1000)]['location'].values
    print(locations)

    #Select meta train
    selected_rows = meta[meta['location'].isin(locations)]
    g=selected_rows.groupby(['split','category_id']).count().reset_index()
    categories =g[(g['split']=='train')&(g['y']>100)]['category_id'].values
    print(categories)
    #Categories is a mapper from order id to cat (categories[index] = category id)

    #Get the mapper from catid to order id :
    cat_order=np.zeros(1000)
    for i, cat in enumerate(categories):
        cat_order[cat]=i

    if save == True:
        with open(folderpath+'train_location_categories_id.npy', 'wb') as f:
            np.save(f, locations)
            np.save(f, categories)
        with open(folderpath+'catorder_to_catid.npy', 'wb') as f:
            np.save(f, locations)
            
        with open(folderpath+'catid_to_catorder.npy', 'wb') as f:
            np.save(f, cat_order)
           
    return locations, categories

def Category_id_order_mapper(catId=None, Catorder=None, id2order=True):
    """
    catID should be list / array / tensor of size batchsize x 1
    read the 0 index from somepath/train_location_categories_id.npy

    id2order == True: take in category id in meta data, return order index
    ==Flase: take in order index, return category id as shown in meta data

    """
    if id2order == True:
        with open(folderpath+'catid_to_catorder.npy', 'rb') as f:
            Catorder = np.load(f)
        return Catorder[catId]
    else: 
        with open(folderpath+'catorder_to_catid.npy', 'rb') as f:
            catId = np.load(f)
        return catId[Catorder]
    



