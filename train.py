import torch
from torch import nn
from torch import optim 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pydoc import locate
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import skimage
from utils import measure_time

    
def train():
    """
    # Creating dataset and dataloader
    with measure_time("Creating datasets"):
        dataset = PYTORCH_DATASET(PATH, PYTORCH_TRANSFORMS)
        dataloader = PYTORCH_DATALOADER(dataset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=4)
    """
    """
    # Fetching class names
    superclass_names = dataset.superclasses
    subclass_names = dataset.subclasses

    # Creating neural network, criterion and optimizer
    net = PYTORCH_MODEL()
    net.to(DEVICE)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    # Training the model
    net.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(dataloader):

            

            #print(data)
            inputs = data["image"].to(DEVICE)
            super_classes = data["superclass"]
            sub_classes = data["subclass"]
            
            outputs = net(inputs)
            
            i += 1
            print(inputs.size())
            print(outputs.size())
            if(i == 100):
                exit(0)
    """

    """
    if DEVICE.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    """


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
                        help="path to the weights of the model")

    # Add more stuff here maybe ?
    args = parser.parse_args()

    global_vars = globals()

    # Checking if gpu is available
    global_vars["DEVICE"] = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu == 'y' else "cpu")


    """
    # Setting global variables
    global_vars["PATH"] = args.path
    global_vars["BATCH_SIZE"] = args.batch
    global_vars["EPOCHS"] = args.epochs
    global_vars["STEPS_PER_EPOCH"] = args.steps

    # Loading custom classes
    global_vars["PYTORCH_DATASET"] = locate("pipeline.pytorch_datasets."
                                          + args.dataset + "." + args.dataset)

    if args.dataloader:
        global_vars["PYTORCH_DATALOADER"] = locate(
            "pipeline.pytorch_dataloader." + args.dataloader + "." + args.dataloader)
    else:
        global_vars["PYTORCH_DATALOADER"] = DataLoader

    global_vars["PYTORCH_TRANSFORMS"] = locate(
        "pipeline.pytorch_transforms." + args.transforms + "." + args.transforms)
    global_vars["PYTORCH_MODEL"] = locate("models." + args.model + "." + args.model)
    """
    train()
