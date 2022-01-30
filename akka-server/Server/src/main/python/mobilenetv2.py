from mediator.server.read_data import ImportData
from generate_partitions import PartitionGenerator
import logging
import argparse
import sys
import asyncio
import numpy as np
import torch 
import random
import torch
import time
import os
import copy
import torchvision
import torch.optim as optim
from torch.optim import Adam
import torch.nn as nn
from torch import device, flatten
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from pathlib import Path
import json
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, dataloader
from mediator.server.custom_dataset import CustomDataset
from PIL import Image

import syft as sy
from syft.workers import websocket_client
from syft.frameworks.torch.fl import utils


class MobileNetV2Post:

    def __init__(self, num_classes, use_pretrained=True):
        
        model_ft, input_size = MobileNetV2Post(12, use_pretrained=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = model_ft.to(device)
        params_to_update = model_ft.parameters()
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        model, val_acc = self.train_model(model=model_ft, criterion=criterion, optimizer=optimizer_ft, num_epochs=5, is_inception=False)

    def initialise_model(self, num_classes, use_pretrained=True):
        model_ft = None
        input_size = 0
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        num_ftrs = 1280
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        return model_ft, input_size

    def train_model(self, model, criterion, optimizer, num_epochs=25):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

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
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(torch.device("cuda" if False else "cpu"))
                    labels = labels.to(torch.device("cuda" if False else "cpu"))

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history