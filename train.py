import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

import torchvision.datasets as datasets
import torchvision.models as models



import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import argparse
import os
import shutil
import time
import random


import copy

from collections import namedtuple




import skimage

from pydoc import locate
from torch.utils.data import Dataset, DataLoader
from utils import measure_time

    
def train():
    pass

def split_dataset(source_path, destination_path, test_dir_name, train_dir_name):
    TRAIN_RATIO = 0.8
    train_dir = os.path.join(destination_path, train_dir_name)
    test_dir = os.path.join(destination_path, test_dir_name)

    # Deleting train and test directories if they exist
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir) 
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Creating directories
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    classes = os.listdir(source_path)

    for c in classes:
        
        class_dir = os.path.join(source_path, c)
        
        images = os.listdir(class_dir)
        
        n_train = int(len(images) * TRAIN_RATIO)
        
        train_images = images[:n_train]
        test_images = images[n_train:]
        
        os.makedirs(os.path.join(train_dir, c), exist_ok = True)
        os.makedirs(os.path.join(test_dir, c), exist_ok = True)

        
        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image) 
            shutil.copyfile(image_src, image_dst)
            
        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image) 
            shutil.copyfile(image_src, image_dst)

def std_mean(train_dir):
    train_data = datasets.ImageFolder(root = train_dir, 
                                    transform = transforms.ToTensor())

    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, label in train_data:
        print(label)
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))

    means /= len(train_data)
    stds /= len(train_data)
    return means, stds

if __name__ == "__main__":
    # Parsing arguments and setting up metadata
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--path",
                        default="data/",
                        help="Path to root directory of the dataset")
    parser.add_argument("--gpu",
                        default="n",
                        help="Use gpu or not (y/n)")
    parser.add_argument("--batch",
                        type=int,
                        default=32,
                        help="Batch size")
    parser.add_argument("--epochs",
                        type=int,
                        default=5,
                        help="Number of epochs")
    parser.add_argument("--steps",
                        type=int,
                        default=10,
                        help="Number of steps per epochs")
    parser.add_argument("--dataset",
                        default="MushroomDataset",
                        help="The pytorch dataset class to be used")
    parser.add_argument("--dataloader",
                        default=None,
                        help="The pytorch dataloader class to be used")
    parser.add_argument("--transforms",
                        default="MushroomTransform",
                        help="The pytorch transforms to be used")
    parser.add_argument("--model",
                        default=None,
                        help="CNN model used")
    parser.add_argument("--weight",
                        default=None,
                        help="Path to the weights of the model")
    parser.add_argument("--split",
                        default=False,
                        help="Whether the dataset is split on disk")
    parser.add_argument("--split-ratio",
                        default=0.8,
                        help="The test/train ratio")
    parser.add_argument("--stats",
                        default=False,
                        help="Whether to compute means and stds or not")

    # Add more stuff here maybe ?
    args = parser.parse_args()

    global_vars = globals()

    # Checking if gpu is available
    global_vars["DEVICE"] = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu == 'y' else "cpu")
    global_vars["DATA_PATH"] = args.path

    # Kiode bizarre - for reproducability
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    data_dir = os.path.join('.', DATA_PATH)
    images_dir = os.path.join(data_dir, 'images')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if args.split:
        with measure_time("Splitting dataset"):
            split_dataset(images_dir, data_dir, 'test', 'train')
    


    if args.stats:
        with measure_time("Computing means and stds"):
            means, std = std_mean(train_dir)
            
        print(f'Calculated means: {means}')
        print(f'Calculated stds: {stds}')

    train()
