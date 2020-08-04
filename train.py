import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms as transforms
import torch.utils.data as data

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

from model import *

    
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
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))

    means /= len(train_data)
    stds /= len(train_data)
    return means, stds
    
def prepare_training(train_dir, test_dir):
#data augmentation: randomly rotating, flipping horizontally and cropping.
    size = 224
    means = [0.4459, 0.4182, 0.3441]
    stds = [0.2210, 0.2137, 0.2109]

    train_transforms = transforms.Compose([
                               transforms.Resize(size),
                               transforms.RandomRotation(5),
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomCrop(size, padding = 10),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = means, 
                                                    std = stds)
                           ])

    test_transforms = transforms.Compose([
                               transforms.Resize(size),
                               transforms.CenterCrop(size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = means, 
                                                    std = stds)
                           ])
    
    train_data = datasets.ImageFolder(root = train_dir, 
                                  transform = train_transforms)

    test_data = datasets.ImageFolder(root = test_dir, 
                                 transform = test_transforms)
                                 
    test_data.classes = [format_label(c) for c in test_data.classes]
    
    #create the validation set
    VALID_RATIO = 0.9

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, 
                                               [n_train_examples, n_valid_examples])
                                               
    valid_data = copy.deepcopy(valid_data) #deepcopy to stop this also changing the training data transforms
    valid_data.dataset.transform = test_transforms 
    
    BATCH_SIZE = 64

    train_iterator = data.DataLoader(train_data, 
                                     shuffle = True, 
                                     batch_size = BATCH_SIZE)

    valid_iterator = data.DataLoader(valid_data, 
                                     batch_size = BATCH_SIZE)

    test_iterator = data.DataLoader(test_data, 
                                    batch_size = BATCH_SIZE)
    #show_img(train_data, test_data)
    
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    
    if MODEL == "resnet18":
        resnet_config = ResNetConfig(block = BasicBlock,
                                n_blocks = [2, 2, 2, 2],
                                channels = [64, 128, 256, 512])
        pretrained_model = models.resnet18(pretrained = True)
    elif MODEL == "resnet34":
        resnet_config = ResNetConfig(block = BasicBlock,
                                n_blocks = [3, 4, 6, 3],
                                channels = [64, 128, 256, 512])
        pretrained_model = models.resnet34(pretrained = True)
    elif MODEL == "resnet50":
        resnet_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 4, 6, 3],
                                channels = [64, 128, 256, 512])
        pretrained_model = models.resnet50(pretrained = True)
    elif MODEL == "resnet101":
        resnet_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 4, 23, 3],
                                channels = [64, 128, 256, 512])
        pretrained_model = models.resnet101(pretrained = True)
    elif MODEL == "resnet152":
        resnet_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 8, 36, 3],
                                channels = [64, 128, 256, 512])
        pretrained_model = models.resnet152(pretrained = True)
    else:
        resnet_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 8, 36, 3],
                                channels = [64, 128, 256, 512])
        pretrained_model = models.resnet152(pretrained = True)
    

    IN_FEATURES = pretrained_model.fc.in_features 
    OUTPUT_DIM = len(test_data.classes)

    fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

    pretrained_model.fc = fc

    model = ResNet(resnet_config, OUTPUT_DIM)
    model.load_state_dict(pretrained_model.state_dict())

    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    
    #find learning rate
    START_LR = 1e-7

    optimizer = optim.Adam(model.parameters(), lr=START_LR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)
    
    #define learning rate finder and run the range test.
    END_LR = 10
    NUM_ITER = 100

    lr_finder = LRFinder(model, optimizer, criterion, device)
    lrs, losses = lr_finder.range_test(train_iterator, END_LR, NUM_ITER)
    
    plot_lr_finder(lrs, losses, skip_start = 30, skip_end = 30)
    
def format_label(label):
    label = label.split('_')[1]#takes only the superclass
    return label

def show_img(train_data, test_data): 
    N_IMAGES = 25

    images, labels = zip(*[(image, label) for image, label in 
                               [train_data[i] for i in range(N_IMAGES)]])


    test_data.classes = [format_label(c) for c in test_data.classes]
    classes = test_data.classes
    plot_images(images, labels, classes)

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image
    
def plot_images(images, labels, classes, normalize = True):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (15, 15))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        label = classes[labels[i]]
        ax.set_title(label)
        ax.axis('off')
    plt.show()


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
    parser.add_argument("--train",
                        default=False,
                        help="Whether the data needs to be trained")

    # Add more stuff here maybe ?
    args = parser.parse_args()

    global_vars = globals()

    # Checking if gpu is available
    global_vars["DEVICE"] = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu == 'y' else "cpu")
    global_vars["DATA_PATH"] = args.path
    global_vars["MODEL"] = args.model

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
            means, stds = std_mean(train_dir)
            
        print(f'Calculated means: {means}')
        print(f'Calculated stds: {stds}')
        global_vars["MEANS"] = means 
        global_vars["STDS"] = stds
        
    if args.train:
        prepare_training(train_dir, test_dir)
    
    
   