import os
import re
import PIL
import sys
import json
import time
import timm
import math
import copy
import torch
import pickle
import geffnet
import logging
import fnmatch
import argparse
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from PIL import Image
from pathlib import Path
from copy import deepcopy
from sklearn import metrics
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as data
from geffnet import create_model
from torch.autograd import Variable
from tqdm import tqdm, tqdm_notebook
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict, defaultdict
from torchvision import transforms, models, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix,accuracy_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    torch.multiprocessing.freeze_support()
    #====Init data directory====
    data_dir = 'C:/Users/AES-VinhToan/Downloads/fruits_dataset'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/val'

    # Define your transforms for the training and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}

    # Using the image datasets and the trainforms, define the data_loader
    # batch_size = 64 for EfficientNet from B0 - B3
    # batch_size = 32 for EfficientNet B4, B5
    # batch_size = 16 for EfficientNet_B6
    # batch_size = 8 for EfficientNet_B7
    # batch_size = 32 for MixNet_s
    batch_size = 32
    data_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=6, pin_memory = True)
                for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    print(dataset_sizes)
    print(device)
    """
    # Label mapping
    with open('cat_to_name-v2.json', 'r') as f:
        cat_to_name = json.load(f)
    """
    ### cat_to_name labels
    f = open('labels-v2.txt','r')
    cat_to_name = f.read()
    print(cat_to_name)
    f.close()

    ### we get the class_to_index in the data_Set but what we really need is the cat_to_names  so we will create
    _ = image_datasets['train'].class_to_idx
    cat_to_name = {_[i]: i for i in list(_.keys())}

        
    # Run this to test the data loader
    images, labels = next(iter(data_loader['val']))
    images.size()


    def showimage(data_loader, number_images, cat_to_name):
        dataiter = iter(data_loader)
        images, labels = dataiter.next()
        images = images.numpy() # convert images to numpy for display
        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(number_images, 4))
        # display 20 images
        for idx in np.arange(number_images):
            ax = fig.add_subplot(2, number_images/2, idx+1, xticks=[], yticks=[])
            img = np.transpose(images[idx])
            plt.imshow(img)
            ax.set_title(cat_to_name[labels.tolist()[idx]])
            

    #### to show some  images
    showimage(data_loader['val'],2,cat_to_name)

    ## Config apply pretraind model 
    ## List model is can use with had already train: efficientnet_es, efficientnet_b3
    use_model ='efficientnet_b3'
    model = create_model(use_model, pretrained=True)

    # Create classifier
    for param in model.parameters():
        param.requires_grad = True

    #num_in_features = 1536 
    # Important set 
    n_classes = 131

    model.classifier = nn.Linear(model.classifier.in_features, n_classes)

    criterion = nn.CrossEntropyLoss()
    #optimizer = Nadam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    #optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9,nesterov=True,weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    model.class_to_idx = image_datasets['train'].class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }


    list(model.class_to_idx.items())


    list(model.idx_to_class.items())


    ### important  fit model with cuda or cpu
    model.to(device)
    
    def train_model(model, criterion, optimizer, scheduler, num_epochs=50, checkpoint = None):
        since = time.time()

        if checkpoint is None:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = math.inf
            best_acc = 0.
        else:
            print(f'Val loss: {checkpoint["best_val_loss"]}, Val accuracy: {checkpoint["best_val_accuracy"]}')
            model.load_state_dict(checkpoint['model_state_dict'])
            best_model_wts = copy.deepcopy(model.state_dict())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_loss = checkpoint['best_val_loss']
            best_acc = checkpoint['best_val_accuracy']

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, (inputs, labels) in enumerate(data_loader[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    if i % 1000 == 999:
                        print('[%d, %d] loss: %.8f' % 
                            (epoch + 1, i, running_loss / (i * inputs.size(0))))

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':                
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':                
                    scheduler.step()
                    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.8f} Acc: {:.8f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print(f'New best model found!')
                    print(f'New record loss: {epoch_loss}, previous record loss: {best_loss}')
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_val_loss': best_loss,
                                'best_val_accuracy': best_acc,
                                'scheduler_state_dict' : scheduler.state_dict(),
                                }, 
                                CHECK_POINT_PATH
                                )
                    print(f'New record loss is SAVED: {epoch_loss}')
               
            print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.8f} Best val loss: {:.8f}'.format(best_acc, best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model, best_loss, best_acc


    # Config Check point path
    CHECK_POINT_PATH = 'D:/Learns/check_point/CHECKPOINT_EFFICIENTNET_ES_TRAIN_SGD.pth'

    try:
        checkpoint = torch.load(CHECK_POINT_PATH)
        print("checkpoint loaded")
    except:
        checkpoint = None
        print("checkpoint not found")
    if checkpoint == None:
        CHECK_POINT_PATH = CHECK_POINT_PATH

    model, best_val_loss, best_val_acc = train_model(model,
                                                    criterion,
                                                    optimizer,
                                                    scheduler,
                                                    num_epochs = 50,
                                                    checkpoint = None
                                                    ) 
                                                    
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_acc,
                'scheduler_state_dict': scheduler.state_dict(),
                }, CHECK_POINT_PATH)
