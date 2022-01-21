'''
python tools
 - Simplified function to create folders
 - Return the filenames into a specific folder
'''

import os
import shutil
import pandas as pd

def safe_make_folder(i, deleteIfExists=False):
    '''Makes a folder if not present'''

    if( deleteIfExists == True ):
        if os.path.exists(i):
            shutil.rmtree(i, ignore_errors=True)
    
    if not os.path.exists(i):
        os.makedirs(i)

def safe_make_list_folders(list_folders):
    """Create a list folders if not present

    Args:
        list_folders (list): list of folders to be created
    """
    for folder in list_folders:
        safe_make_folder(folder)

def get_filenames(directory):
    """Return filenames in directory

    Args:
        directory (str): directory path

    Returns:
        list: File list inside directory
    """
       
    return sorted(os.listdir(directory))  

def get_filenames_dataset(params,dataset='test'):
    """get the filenames used on a dataset. 

    Args:
        params (params): params configuration file
        dataset (str, optional): dataset type. Defaults to 'test'.

    Returns:
        [list]: list with filenames
    """
    dataset_path = params.dataset_path
    csv_file = dataset_path + '/' + dataset + '.csv'
    df = pd.read_csv(csv_file)
    filenames = df['filepath'].tolist()   
    return filenames