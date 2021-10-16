import argparse
import glob
import os
import random
import shutil
import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    
    files = os.listdir(data_dir)

    try:
        os.makedirs(f"{data_dir}/train/")
        os.makedirs(f"{data_dir}/val/")
        os.makedirs(f"{data_dir}/test/")
    except Exception as e:
        print(e)
        pass

    random.shuffle(files)

    for file in files[:70]:
        source = f"{data_dir}/{file}"
        destination = f"{data_dir}/train/{file}"
        shutil.move(source, destination)


    for file in files[70:80]:
        source = f"{data_dir}/{file}"
        destination = f"{data_dir}/val/{file}"
        shutil.move(source, destination)


    for file in files[80:]:
        source = f"{data_dir}/{file}"
        destination = f"{data_dir}/test/{file}"
        shutil.move(source, destination)
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)